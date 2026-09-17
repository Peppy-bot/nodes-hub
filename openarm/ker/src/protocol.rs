// KER wire protocol: pure byte parsing, no transport. The device (an M5Stack
// CoreS3) speaks a small framed protocol over USB vendor mode or the ESP32's
// serial device:
//
// - Host commands are single bytes (`CMD_*`).
// - The PING response (`PING_HEADER`) carries device metadata and a
//   self-describing schema: firmware[16] hardware[16] updated[12] (NUL-padded
//   utf8), a field count, then per field a key[16], a type id and an element
//   count.
// - Stream packets (`STREAM_HEADER`) carry the schema's fields packed
//   little-endian, followed by one XOR checksum byte over the payload.
//
// Parsing is staged parse-don't-validate: `Schema::parse_ping` turns the
// handshake bytes into a typed [`Schema`] once, [`FrameLayout::try_new`]
// resolves the field offsets this node consumes once (failing loudly on an
// incompatible schema), and per packet only `Deframer` + [`FrameLayout::parse`]
// run, both infallible on checksum-verified payloads.
//
// Under `cfg(test)` this module also carries `fixtures`, the device-side byte
// builders this crate's tests share (they need the field widths above).

use std::fmt;

pub const CMD_PING: u8 = 0x00;
pub const CMD_STANDBY: u8 = 0x01;
pub const CMD_STREAM: u8 = 0x02;
const PING_HEADER: [u8; 2] = [0xA5, 0x50];
const STREAM_HEADER: [u8; 2] = [0xA5, 0x5A];

const FW_LEN: usize = 16;
const HW_LEN: usize = 16;
const UPDATED_LEN: usize = 12;
const KEY_LEN: usize = 16;
/// key[16] + type_id + count
const FIELD_ENTRY_LEN: usize = KEY_LEN + 2;
/// header + metadata strings + field count
const PING_FIXED_LEN: usize = 2 + FW_LEN + HW_LEN + UPDATED_LEN + 1;
/// Bytes the response carries per sensor after its field table: the sensor's
/// angle and one status byte, which the firmware appends to every PING reply.
const SNAPSHOT_ENTRY_LEN: usize = 5;
/// The longest response the device can describe: every field a `u8` count can
/// name, and a snapshot entry for every channel one of them can carry.
pub(crate) const MAX_PING_RESPONSE_LEN: usize =
    PING_FIXED_LEN + u8::MAX as usize * FIELD_ENTRY_LEN + u8::MAX as usize * SNAPSHOT_ENTRY_LEN;
/// The field every stream packet carries.
const REQUIRED_FIELD: &str = "angles";

/// What one candidate header position turned out to be.
enum Candidate {
    Parsed { schema: Schema, consumed: usize },
    NeedMore,
    NotAResponse,
}

#[derive(Debug)]
pub enum ProtocolError {
    MissingField(&'static str),
    WrongFieldType {
        key: &'static str,
        expected: &'static str,
    },
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(key) => write!(f, "schema is missing the '{key}' field"),
            Self::WrongFieldType { key, expected } => {
                write!(f, "schema field '{key}' is not {expected}")
            }
        }
    }
}

impl std::error::Error for ProtocolError {}

/// A stream field's element type, from the schema's type id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldType {
    U32,
    U16,
    U8,
    I32,
    I16,
    F32,
    Bool,
}

impl FieldType {
    fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(Self::U32),
            1 => Some(Self::U16),
            2 => Some(Self::U8),
            3 => Some(Self::I32),
            4 => Some(Self::I16),
            5 => Some(Self::F32),
            6 => Some(Self::Bool),
            _ => None,
        }
    }

    fn size(self) -> usize {
        match self {
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U16 | Self::I16 => 2,
            Self::U8 | Self::Bool => 1,
        }
    }
}

/// One field of the device's stream packet layout.
#[derive(Debug, Clone)]
struct FieldDesc {
    key: String,
    ty: FieldType,
    count: usize,
}

/// Device identity strings from the PING response.
#[derive(Debug, Clone)]
pub struct Metadata {
    pub firmware: String,
    pub hardware: String,
    pub updated: String,
}

impl Metadata {
    /// Whether these strings read as a device's own: a firmware and a
    /// hardware version, printable and NUL-padded. Stream payload decoded at
    /// a false header fails this, which is what keeps noise out of a
    /// handshake.
    fn is_plausible(&self) -> bool {
        [&self.firmware, &self.hardware]
            .into_iter()
            .all(|field| !field.is_empty() && field.chars().all(|c| c.is_ascii_graphic()))
    }
}

/// The device's self-described stream layout, parsed once at handshake.
#[derive(Debug)]
pub struct Schema {
    pub metadata: Metadata,
    fields: Vec<FieldDesc>,
}

/// Outcome of feeding handshake bytes to [`Schema::parse_ping`].
#[derive(Debug)]
pub enum PingParse {
    /// No complete response buffered yet; keep the buffer and read more.
    NeedMore,
    /// Parsed; `consumed` bytes (up to and including the response) are spent.
    Parsed { schema: Schema, consumed: usize },
}

impl Schema {
    /// Scan `buf` for a PING response and parse it. Bytes before the header are
    /// ignored (the device may still be streaming), and a header byte pair that
    /// turns out to be stream payload is skipped so the response behind it is
    /// still found.
    pub fn parse_ping(buf: &[u8]) -> PingParse {
        let mut from = 0;
        while let Some(offset) = find_header(&buf[from..], PING_HEADER) {
            let start = from + offset;
            // A candidate that overruns the buffer is undecidable, and a
            // complete response can sit inside the length it claims, so the
            // scan carries on past both it and outright noise.
            if let Candidate::Parsed { schema, consumed } = Self::parse_at(buf, start) {
                return PingParse::Parsed { schema, consumed };
            }
            from = start + 1;
        }
        PingParse::NeedMore
    }

    /// Read one candidate response at `start`.
    fn parse_at(buf: &[u8], start: usize) -> Candidate {
        let b = &buf[start..];
        if b.len() < PING_FIXED_LEN {
            return Candidate::NeedMore;
        }
        let firmware = padded_str(&b[2..2 + FW_LEN]);
        let hardware = padded_str(&b[2 + FW_LEN..2 + FW_LEN + HW_LEN]);
        let updated = padded_str(&b[2 + FW_LEN + HW_LEN..2 + FW_LEN + HW_LEN + UPDATED_LEN]);
        let field_count = b[PING_FIXED_LEN - 1] as usize;
        if b.len() < PING_FIXED_LEN + field_count * FIELD_ENTRY_LEN {
            return Candidate::NeedMore;
        }
        let mut fields = Vec::with_capacity(field_count);
        for i in 0..field_count {
            let entry = &b[PING_FIXED_LEN + i * FIELD_ENTRY_LEN..];
            let type_id = entry[KEY_LEN];
            // A type id the schema never uses means these bytes are stream
            // payload that happened to carry the header.
            let Some(ty) = FieldType::from_id(type_id) else {
                return Candidate::NotAResponse;
            };
            fields.push(FieldDesc {
                key: padded_str(&entry[..KEY_LEN]),
                ty,
                count: entry[KEY_LEN + 1] as usize,
            });
        }
        let metadata = Metadata {
            firmware,
            hardware,
            updated,
        };
        // Stream payload carrying the header parses this far whenever its
        // field bytes happen to read as a table, including an empty one, so
        // what tells a response from noise is metadata a device would send.
        if !metadata.is_plausible() {
            return Candidate::NotAResponse;
        }
        // The reply ends with one snapshot entry per angle channel. The
        // firmware writes it in packets of its own, so the response counts as
        // read at the end of the table and the block is consumed as far as it
        // has arrived; a tail landing later reaches the deframer, which
        // resyncs on the next packet header.
        let table_end = start + PING_FIXED_LEN + field_count * FIELD_ENTRY_LEN;
        let angles = fields
            .iter()
            .find(|field| field.key == REQUIRED_FIELD)
            .map_or(0, |field| field.count);
        let snapshot = (angles * SNAPSHOT_ENTRY_LEN).min(buf.len() - table_end);
        Candidate::Parsed {
            schema: Schema { metadata, fields },
            consumed: table_end + snapshot,
        }
    }
}

/// One decoded stream packet, still device-shaped: raw channels in degrees.
#[derive(Debug, PartialEq)]
pub struct KerFrame {
    /// All encoder channels (deg), CH01 at index 0.
    pub angles_deg: Vec<f32>,
}

/// Byte offsets of the fields this node consumes, resolved from a [`Schema`]
/// once at handshake. `angles` is required; every other field the device
/// streams is skipped by its packed size.
#[derive(Debug)]
pub struct FrameLayout {
    payload_len: usize,
    angles_at: usize,
    angle_count: usize,
}

impl FrameLayout {
    pub fn try_new(schema: &Schema) -> Result<Self, ProtocolError> {
        let mut offset = 0;
        let mut angles = None;
        for field in &schema.fields {
            match (field.key.as_str(), field.ty) {
                // The first match wins, as the response scan reads it.
                (REQUIRED_FIELD, FieldType::F32) if angles.is_none() => {
                    angles = Some((offset, field.count))
                }
                (REQUIRED_FIELD, _) => {
                    return Err(ProtocolError::WrongFieldType {
                        key: REQUIRED_FIELD,
                        expected: "f32",
                    });
                }
                _ => {}
            }
            offset += field.ty.size() * field.count;
        }
        let (angles_at, angle_count) = angles.ok_or(ProtocolError::MissingField(REQUIRED_FIELD))?;
        Ok(Self {
            payload_len: offset,
            angles_at,
            angle_count,
        })
    }

    /// Packed byte length of the payload this layout decodes. The deframer is
    /// sized from here, so the length it delivers and the length decoded against
    /// are one value rather than two derivations of the same sum.
    pub fn payload_len(&self) -> usize {
        self.payload_len
    }

    pub fn angle_count(&self) -> usize {
        self.angle_count
    }

    /// Decode one checksum-verified payload of exactly [`Self::payload_len`]
    /// bytes, which is what the deframer this layout sized delivers.
    pub fn parse(&self, payload: &[u8]) -> KerFrame {
        debug_assert_eq!(
            payload.len(),
            self.payload_len,
            "the deframer delivers exactly the payload this layout decodes"
        );
        let angles_deg = (0..self.angle_count)
            .map(|i| {
                let at = self.angles_at + i * 4;
                f32::from_le_bytes(payload[at..at + 4].try_into().expect("4 bytes"))
            })
            .collect();
        KerFrame { angles_deg }
    }
}

/// XOR of every payload byte: the device's stream packet checksum.
fn xor_checksum(payload: &[u8]) -> u8 {
    payload.iter().fold(0, |acc, b| acc ^ b)
}

/// Splits a byte stream into checksum-verified stream packet payloads.
/// Corruption resyncs by discarding only the matched header, so a valid packet
/// immediately after a false header match is still found.
pub struct Deframer {
    buf: Vec<u8>,
    payload_len: usize,
}

/// A stream packet whose checksum did not match; the frame is discarded.
#[derive(Debug, PartialEq, Eq)]
pub struct BadChecksum;

impl Deframer {
    pub fn new(payload_len: usize) -> Self {
        Self {
            buf: Vec::new(),
            payload_len,
        }
    }

    pub fn push(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Bytes held back for a packet that has yet to complete, which is what
    /// proves garbage cannot accumulate.
    #[cfg(test)]
    fn buffered(&self) -> usize {
        self.buf.len()
    }

    /// The next payload, `Some(Err)` for a corrupt frame, or `None` when no
    /// complete packet is buffered.
    pub fn next_payload(&mut self) -> Option<Result<Vec<u8>, BadChecksum>> {
        let Some(start) = find_header(&self.buf, STREAM_HEADER) else {
            // Nothing useful before a possible header first byte at the
            // tail; drop the rest so garbage cannot accumulate.
            let keep = usize::from(self.buf.last() == Some(&STREAM_HEADER[0]));
            self.buf.drain(..self.buf.len() - keep);
            return None;
        };
        self.buf.drain(..start);
        let packet_len = 2 + self.payload_len + 1;
        if self.buf.len() < packet_len {
            return None;
        }
        let payload = &self.buf[2..2 + self.payload_len];
        if xor_checksum(payload) == self.buf[packet_len - 1] {
            let payload = payload.to_vec();
            self.buf.drain(..packet_len);
            return Some(Ok(payload));
        }
        // False or corrupted header: skip it so the next call rescans from the
        // following byte, keeping a real packet right behind it reachable.
        self.buf.drain(..2);
        Some(Err(BadChecksum))
    }
}

fn find_header(buf: &[u8], header: [u8; 2]) -> Option<usize> {
    buf.windows(2).position(|w| w == header)
}

fn padded_str(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .trim_end_matches('\0')
        .to_string()
}

#[cfg(test)]
pub(crate) mod fixtures {
    //! Device-side byte builders shared by this crate's tests.

    use super::*;

    /// A schema key, NUL-padded the way the device writes one.
    pub(crate) fn padded_key(key: &str) -> Vec<u8> {
        padded(key, KEY_LEN)
    }

    fn padded(s: &str, len: usize) -> Vec<u8> {
        let mut v = s.as_bytes().to_vec();
        assert!(v.len() <= len);
        v.resize(len, 0);
        v
    }

    /// A ping response shaped like firmware 2.0.0's: the schema it registers,
    /// followed by the snapshot block every reply carries.
    pub(crate) fn ping_response(channels: u8) -> Vec<u8> {
        ping_response_for("2.0.0", channels)
    }

    /// The same response for a chosen hardware version.
    pub(crate) fn ping_response_for(hardware: &str, channels: u8) -> Vec<u8> {
        let mut v = PING_HEADER.to_vec();
        v.extend(padded("2.0.0", FW_LEN));
        v.extend(padded(hardware, HW_LEN));
        v.extend(padded("2026-06-22", UPDATED_LEN));
        v.push(3);
        for (key, type_id, count) in [
            ("timestamp", 0u8, 1u8),
            ("angles", 5, channels),
            ("errors", 6, channels),
        ] {
            v.extend(padded(key, KEY_LEN));
            v.push(type_id);
            v.push(count);
        }
        // One angle and one status byte per channel, as the firmware appends.
        for channel in 0..channels {
            v.extend((channel as f32).to_le_bytes());
            v.push(1);
        }
        v
    }

    /// A stream packet for the reference schema: timestamp, the angles, and
    /// one error flag per channel, with a valid checksum.
    pub(crate) fn stream_packet(timestamp: u32, angles: &[f32]) -> Vec<u8> {
        let mut payload = timestamp.to_le_bytes().to_vec();
        for a in angles {
            payload.extend(a.to_le_bytes());
        }
        payload.extend(std::iter::repeat_n(0u8, angles.len()));
        let mut packet = STREAM_HEADER.to_vec();
        packet.push(xor_checksum(&payload));
        packet.splice(2..2, payload);
        packet
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;

    /// The layout the reader builds; deframer sizing goes through it here too,
    /// so the tests exercise the single derivation production uses.
    fn layout_of(schema: &Schema) -> FrameLayout {
        FrameLayout::try_new(schema).expect("layout")
    }

    fn reference_schema(channels: u8) -> Schema {
        match Schema::parse_ping(&ping_response(channels)) {
            PingParse::Parsed { schema, .. } => schema,
            other => panic!("expected parse, got {other:?}"),
        }
    }

    #[test]
    fn ping_round_trips_metadata_and_fields() {
        let response = ping_response(16);
        let PingParse::Parsed { schema, consumed } = Schema::parse_ping(&response) else {
            panic!("expected parse");
        };
        assert_eq!(consumed, response.len());
        assert_eq!(schema.metadata.firmware, "2.0.0");
        assert_eq!(schema.metadata.hardware, "2.0.0");
        assert_eq!(schema.metadata.updated, "2026-06-22");
        assert_eq!(schema.fields.len(), 3);
        assert_eq!(schema.fields[1].key, "angles");
        assert_eq!(schema.fields[1].ty, FieldType::F32);
        assert_eq!(schema.fields[1].count, 16);
        let layout = layout_of(&schema);
        // timestamp, sixteen angles, sixteen error flags.
        assert_eq!(layout.payload_len(), 4 + 16 * 4 + 16);
    }

    #[test]
    fn a_reply_whose_snapshot_block_is_still_in_flight_still_connects() {
        // The firmware sends the reply in packets of its own, so the block
        // after the schema can lag. Everything through the table is what the
        // node needs.
        let response = ping_response(16);
        let table_end = response.len() - 16 * SNAPSHOT_ENTRY_LEN;
        for arrived in [table_end, table_end + 7, response.len()] {
            let PingParse::Parsed { schema, consumed } = Schema::parse_ping(&response[..arrived])
            else {
                panic!("a complete table parses with {arrived} bytes in hand");
            };
            assert_eq!(schema.metadata.hardware, "2.0.0");
            assert_eq!(
                consumed, arrived,
                "the block is consumed as far as it arrived"
            );
        }
    }

    #[test]
    fn ping_needs_more_on_every_truncation() {
        let response = ping_response(16);
        let table_end = response.len() - 16 * SNAPSHOT_ENTRY_LEN;
        for len in 0..table_end {
            assert!(
                matches!(Schema::parse_ping(&response[..len]), PingParse::NeedMore),
                "truncation at {len} must ask for more"
            );
        }
    }

    #[test]
    fn a_false_ping_header_reading_as_an_empty_table_is_skipped() {
        // Zero bytes are the common case in stream payload: a released
        // channel, a small timestamp's high bytes, a cleared error flag. Such
        // a candidate carries no version strings, so it reads as payload.
        let mut buf = PING_HEADER.to_vec();
        buf.extend([0x00; PING_FIXED_LEN * 2]);
        let response = ping_response(16);
        buf.extend(&response);
        let PingParse::Parsed { schema, consumed } = Schema::parse_ping(&buf) else {
            panic!("expected the real response to parse");
        };
        assert_eq!(schema.metadata.hardware, "2.0.0");
        assert_eq!(consumed, buf.len());
    }

    #[test]
    fn a_candidate_claiming_more_bytes_than_arrived_does_not_hide_a_response() {
        // The count byte sits at PING_FIXED_LEN - 1 of the candidate: a false
        // header claiming 255 fields needs 4637 bytes, far more than arrive,
        // so it is undecidable. The complete response behind it must still be
        // found.
        let mut buf = PING_HEADER.to_vec();
        buf.extend([0x01; PING_FIXED_LEN - 3]);
        buf.push(0xFF);
        buf.extend(ping_response(16));
        let PingParse::Parsed { schema, .. } = Schema::parse_ping(&buf) else {
            panic!("expected the real response to parse");
        };
        assert_eq!(schema.metadata.hardware, "2.0.0");
    }

    #[test]
    fn a_false_ping_header_in_stream_bytes_is_skipped() {
        // Stream payload that happens to carry the ping header, with metadata
        // that reads like a device's but a type id the schema never uses: the
        // response behind it must still be found.
        let mut buf = PING_HEADER.to_vec();
        buf.extend([b'A'; PING_FIXED_LEN - 3]);
        buf.push(1);
        buf.extend(fixtures::padded_key("angles"));
        buf.push(7);
        buf.push(16);
        let response = ping_response(16);
        buf.extend(&response);
        let PingParse::Parsed { schema, consumed } = Schema::parse_ping(&buf) else {
            panic!("expected the real response to parse");
        };
        assert_eq!(schema.metadata.hardware, "2.0.0");
        assert_eq!(consumed, buf.len());
    }

    #[test]
    fn a_ping_response_behind_garbage_keeps_its_metadata() {
        let mut buf = vec![0x11, 0xA5, 0x22];
        buf.extend(ping_response(8));
        let PingParse::Parsed { schema, consumed } = Schema::parse_ping(&buf) else {
            panic!("expected parse");
        };
        assert_eq!(schema.metadata.firmware, "2.0.0");
        assert_eq!(schema.metadata.hardware, "2.0.0");
        assert_eq!(consumed, buf.len());
    }

    #[test]
    fn layout_decodes_a_packet_exactly() {
        let schema = reference_schema(3);
        let layout = layout_of(&schema);
        assert_eq!(layout.angle_count(), 3);

        let packet = stream_packet(7, &[10.0, -20.5, 30.25]);
        let mut deframer = Deframer::new(layout.payload_len());
        deframer.push(&packet);
        let payload = deframer.next_payload().expect("one frame").expect("valid");
        assert_eq!(
            layout.parse(&payload),
            KerFrame {
                angles_deg: vec![10.0, -20.5, 30.25],
            }
        );
    }

    #[test]
    fn layout_requires_f32_angles() {
        let schema = reference_schema(3);
        let missing = Schema {
            metadata: schema.metadata.clone(),
            fields: schema
                .fields
                .iter()
                .filter(|f| f.key != "angles")
                .cloned()
                .collect(),
        };
        assert!(matches!(
            FrameLayout::try_new(&missing),
            Err(ProtocolError::MissingField("angles"))
        ));
        let wrong_type = Schema {
            metadata: schema.metadata.clone(),
            fields: schema
                .fields
                .iter()
                .cloned()
                .map(|mut f| {
                    if f.key == "angles" {
                        f.ty = FieldType::I16;
                    }
                    f
                })
                .collect(),
        };
        assert!(matches!(
            FrameLayout::try_new(&wrong_type),
            Err(ProtocolError::WrongFieldType { key: "angles", .. })
        ));
    }

    #[test]
    fn a_schema_of_angles_alone_decodes() {
        let schema = Schema {
            metadata: reference_schema(1).metadata,
            fields: vec![FieldDesc {
                key: "angles".into(),
                ty: FieldType::F32,
                count: 2,
            }],
        };
        let layout = layout_of(&schema);
        let frame = layout.parse(&[0, 0, 128, 63, 0, 0, 0, 64]);
        assert_eq!(frame.angles_deg, vec![1.0, 2.0]);
    }

    #[test]
    fn layout_skips_unknown_fields_by_size() {
        // A future firmware inserts an unknown field before the angles.
        let schema = Schema {
            metadata: reference_schema(1).metadata,
            fields: vec![
                FieldDesc {
                    key: "battery_mv".into(),
                    ty: FieldType::U16,
                    count: 1,
                },
                FieldDesc {
                    key: "angles".into(),
                    ty: FieldType::F32,
                    count: 1,
                },
            ],
        };
        let layout = layout_of(&schema);
        let mut payload = 500u16.to_le_bytes().to_vec();
        payload.extend(90.0f32.to_le_bytes());
        assert_eq!(layout.parse(&payload).angles_deg, vec![90.0]);
    }

    #[test]
    fn a_reordered_schema_deframes_and_decodes_at_the_same_length() {
        // The reader sizes the deframer from the layout, so a firmware that
        // orders or types its fields differently from the reference still hands
        // `parse` exactly the bytes it decodes.
        let schema = Schema {
            metadata: reference_schema(1).metadata,
            fields: vec![
                FieldDesc {
                    key: "charging".into(),
                    ty: FieldType::Bool,
                    count: 1,
                },
                FieldDesc {
                    key: "angles".into(),
                    ty: FieldType::F32,
                    count: 2,
                },
                FieldDesc {
                    key: "temperature_c".into(),
                    ty: FieldType::I16,
                    count: 1,
                },
                // Fields this node does not consume still occupy their bytes.
                FieldDesc {
                    key: "spare".into(),
                    ty: FieldType::U32,
                    count: 1,
                },
            ],
        };
        let layout = layout_of(&schema);
        assert_eq!(layout.payload_len(), 1 + 2 * 4 + 2 + 4);

        let mut payload = vec![1u8];
        payload.extend(1.5f32.to_le_bytes());
        payload.extend((-2.5f32).to_le_bytes());
        payload.extend((-7i16).to_le_bytes());
        payload.extend(0u32.to_le_bytes());
        let mut packet = STREAM_HEADER.to_vec();
        packet.push(xor_checksum(&payload));
        packet.splice(2..2, payload);

        let mut deframer = Deframer::new(layout.payload_len());
        deframer.push(&packet);
        let delivered = deframer.next_payload().expect("one frame").expect("valid");
        assert_eq!(
            delivered.len(),
            layout.payload_len(),
            "the deframer must deliver exactly what the layout decodes"
        );
        assert_eq!(
            layout.parse(&delivered),
            KerFrame {
                angles_deg: vec![1.5, -2.5],
            }
        );
    }

    #[test]
    fn deframer_reassembles_byte_at_a_time_delivery() {
        let schema = reference_schema(2);
        let layout = layout_of(&schema);
        let mut deframer = Deframer::new(layout.payload_len());
        let packet = stream_packet(1, &[1.0, 2.0]);
        for (i, byte) in packet.iter().enumerate() {
            deframer.push(&[*byte]);
            if i < packet.len() - 1 {
                assert!(deframer.next_payload().is_none(), "byte {i} is not a frame");
            }
        }
        assert!(deframer.next_payload().expect("frame").is_ok());
    }

    #[test]
    fn deframer_drops_a_corrupt_frame_and_resyncs() {
        let schema = reference_schema(2);
        let layout = layout_of(&schema);
        let mut deframer = Deframer::new(layout.payload_len());
        let mut corrupted = stream_packet(1, &[1.0, 2.0]);
        let last = corrupted.len() - 1;
        corrupted[last] ^= 0xFF;
        deframer.push(&corrupted);
        deframer.push(&stream_packet(2, &[3.0, 4.0]));
        assert_eq!(deframer.next_payload(), Some(Err(BadChecksum)));
        let payload = deframer.next_payload().expect("frame").expect("valid");
        assert_eq!(layout.parse(&payload).angles_deg, vec![3.0, 4.0]);
    }

    #[test]
    fn deframer_skips_leading_garbage_and_bounds_its_buffer() {
        let schema = reference_schema(2);
        let layout = layout_of(&schema);
        let mut deframer = Deframer::new(layout.payload_len());
        deframer.push(&[0x00, 0xA5, 0x00, 0xFF]);
        assert!(deframer.next_payload().is_none());
        assert_eq!(deframer.buffered(), 0, "garbage must not accumulate");
        deframer.push(&stream_packet(3, &[0.5, -0.5]));
        let payload = deframer.next_payload().expect("frame").expect("valid");
        assert_eq!(layout.parse(&payload).angles_deg, vec![0.5, -0.5]);
    }

    #[test]
    fn deframer_keeps_a_trailing_possible_header_byte() {
        let schema = reference_schema(2);
        let layout = layout_of(&schema);
        let mut deframer = Deframer::new(layout.payload_len());
        let packet = stream_packet(4, &[1.0, 1.0]);
        deframer.push(&[0x33, STREAM_HEADER[0]]);
        assert!(deframer.next_payload().is_none());
        // The 0xA5 tail must survive so a packet split right after it still parses.
        deframer.push(&packet[1..]);
        assert!(deframer.next_payload().expect("frame").is_ok());
    }
}
