// KER wire protocol: pure byte parsing, no transport. The device (an M5Stack
// CoreS3) speaks a small framed protocol over USB vendor mode or serial CDC:
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

use std::fmt;

pub const CMD_PING: u8 = 0x00;
pub const CMD_STANDBY: u8 = 0x01;
pub const PING_HEADER: [u8; 2] = [0xA5, 0x50];
pub const STREAM_HEADER: [u8; 2] = [0xA5, 0x5A];

const FW_LEN: usize = 16;
const HW_LEN: usize = 16;
const UPDATED_LEN: usize = 12;
const KEY_LEN: usize = 16;
/// key[16] + type_id + count
const FIELD_ENTRY_LEN: usize = KEY_LEN + 2;
/// header + metadata strings + field count
const PING_FIXED_LEN: usize = 2 + FW_LEN + HW_LEN + UPDATED_LEN + 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    UnknownTypeId(u8),
    MissingField(&'static str),
    WrongFieldType {
        key: &'static str,
        expected: &'static str,
    },
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTypeId(id) => write!(f, "schema field with unknown type id {id}"),
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
pub enum FieldType {
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

    pub fn size(self) -> usize {
        match self {
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U16 | Self::I16 => 2,
            Self::U8 | Self::Bool => 1,
        }
    }
}

/// One field of the device's stream packet layout.
#[derive(Debug, Clone)]
pub struct FieldDesc {
    pub key: String,
    pub ty: FieldType,
    pub count: usize,
}

/// Device identity strings from the PING response.
#[derive(Debug, Clone)]
pub struct Metadata {
    pub firmware: String,
    pub hardware: String,
    pub updated: String,
}

/// The device's self-described stream layout, parsed once at handshake.
#[derive(Debug, Clone)]
pub struct Schema {
    pub metadata: Metadata,
    pub fields: Vec<FieldDesc>,
}

/// Outcome of feeding handshake bytes to [`Schema::parse_ping`].
#[derive(Debug)]
pub enum PingParse {
    /// No complete response buffered yet; keep the buffer and read more.
    NeedMore,
    /// Parsed; `consumed` bytes (up to and including the response) are spent.
    Parsed {
        schema: Schema,
        consumed: usize,
    },
    Invalid(ProtocolError),
}

impl Schema {
    /// Scan `buf` for a PING response and parse it. Bytes before the header are
    /// ignored (the device may still be flushing stream packets).
    pub fn parse_ping(buf: &[u8]) -> PingParse {
        let Some(start) = find_header(buf, PING_HEADER) else {
            return PingParse::NeedMore;
        };
        let b = &buf[start..];
        if b.len() < PING_FIXED_LEN {
            return PingParse::NeedMore;
        }
        let firmware = padded_str(&b[2..2 + FW_LEN]);
        let hardware = padded_str(&b[2 + FW_LEN..2 + FW_LEN + HW_LEN]);
        let updated = padded_str(&b[2 + FW_LEN + HW_LEN..2 + FW_LEN + HW_LEN + UPDATED_LEN]);
        let field_count = b[PING_FIXED_LEN - 1] as usize;
        if b.len() < PING_FIXED_LEN + field_count * FIELD_ENTRY_LEN {
            return PingParse::NeedMore;
        }
        let mut fields = Vec::with_capacity(field_count);
        for i in 0..field_count {
            let entry = &b[PING_FIXED_LEN + i * FIELD_ENTRY_LEN..];
            let type_id = entry[KEY_LEN];
            let Some(ty) = FieldType::from_id(type_id) else {
                return PingParse::Invalid(ProtocolError::UnknownTypeId(type_id));
            };
            fields.push(FieldDesc {
                key: padded_str(&entry[..KEY_LEN]),
                ty,
                count: entry[KEY_LEN + 1] as usize,
            });
        }
        PingParse::Parsed {
            schema: Schema {
                metadata: Metadata {
                    firmware,
                    hardware,
                    updated,
                },
                fields,
            },
            consumed: start + PING_FIXED_LEN + field_count * FIELD_ENTRY_LEN,
        }
    }
}

/// One decoded stream packet, still device-shaped: raw channels in degrees.
#[derive(Debug, Clone, PartialEq)]
pub struct KerFrame {
    pub timestamp: u32,
    /// All encoder channels (deg), CH01 at index 0.
    pub angles_deg: Vec<f32>,
    pub encoder_value: i64,
    pub encoder_button: bool,
}

/// Byte offsets of the fields this node consumes, resolved from a [`Schema`]
/// once at handshake. `angles` is required; the rest degrade to defaults so a
/// newer firmware can drop or retype them without breaking arm streaming
/// (the reader refuses button-gated engage separately when the button is absent).
#[derive(Debug, Clone)]
pub struct FrameLayout {
    payload_len: usize,
    angles_at: usize,
    angle_count: usize,
    timestamp_at: Option<usize>,
    encoder_value_at: Option<(usize, FieldType)>,
    encoder_button_at: Option<usize>,
}

impl FrameLayout {
    pub fn try_new(schema: &Schema) -> Result<Self, ProtocolError> {
        let mut offset = 0;
        let mut angles = None;
        let mut timestamp_at = None;
        let mut encoder_value_at = None;
        let mut encoder_button_at = None;
        for field in &schema.fields {
            match (field.key.as_str(), field.ty) {
                ("angles", FieldType::F32) => angles = Some((offset, field.count)),
                ("angles", _) => {
                    return Err(ProtocolError::WrongFieldType {
                        key: "angles",
                        expected: "f32",
                    });
                }
                ("timestamp", FieldType::U32) if field.count == 1 => {
                    timestamp_at = Some(offset);
                }
                ("encoder_value", ty) if field.count == 1 && ty != FieldType::F32 => {
                    encoder_value_at = Some((offset, ty));
                }
                ("encoder_button", FieldType::Bool | FieldType::U8) if field.count == 1 => {
                    encoder_button_at = Some(offset);
                }
                _ => {}
            }
            offset += field.ty.size() * field.count;
        }
        let (angles_at, angle_count) = angles.ok_or(ProtocolError::MissingField("angles"))?;
        Ok(Self {
            payload_len: offset,
            angles_at,
            angle_count,
            timestamp_at,
            encoder_value_at,
            encoder_button_at,
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

    /// Whether the schema carries the thumb button (required for toggle engage).
    pub fn has_button(&self) -> bool {
        self.encoder_button_at.is_some()
    }

    /// Decode one checksum-verified payload of exactly [`Self::payload_len`]
    /// bytes, which is what the deframer this layout sized delivers.
    pub fn parse(&self, payload: &[u8]) -> KerFrame {
        let angles_deg = (0..self.angle_count)
            .map(|i| {
                let at = self.angles_at + i * 4;
                f32::from_le_bytes(payload[at..at + 4].try_into().expect("4 bytes"))
            })
            .collect();
        let timestamp = self
            .timestamp_at
            .map(|at| u32::from_le_bytes(payload[at..at + 4].try_into().expect("4 bytes")))
            .unwrap_or(0);
        let encoder_value = self
            .encoder_value_at
            .map(|(at, ty)| read_int(payload, at, ty))
            .unwrap_or(0);
        let encoder_button = self
            .encoder_button_at
            .map(|at| payload[at] != 0)
            .unwrap_or(false);
        KerFrame {
            timestamp,
            angles_deg,
            encoder_value,
            encoder_button,
        }
    }
}

fn read_int(payload: &[u8], at: usize, ty: FieldType) -> i64 {
    let le = |n: usize| -> [u8; 8] {
        let mut b = [0u8; 8];
        b[..n].copy_from_slice(&payload[at..at + n]);
        b
    };
    match ty {
        FieldType::U32 => u32::from_le_bytes(le(4)[..4].try_into().expect("4 bytes")) as i64,
        FieldType::U16 => u16::from_le_bytes(le(2)[..2].try_into().expect("2 bytes")) as i64,
        FieldType::U8 | FieldType::Bool => payload[at] as i64,
        FieldType::I32 => i32::from_le_bytes(le(4)[..4].try_into().expect("4 bytes")) as i64,
        FieldType::I16 => i16::from_le_bytes(le(2)[..2].try_into().expect("2 bytes")) as i64,
        FieldType::F32 => unreachable!("layout never selects f32 for an integer field"),
    }
}

/// XOR of every payload byte: the device's stream packet checksum.
pub fn xor_checksum(payload: &[u8]) -> u8 {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
mod tests {
    use super::*;

    /// The layout the reader builds; deframer sizing goes through it here too,
    /// so the tests exercise the single derivation production uses.
    fn layout_of(schema: &Schema) -> FrameLayout {
        FrameLayout::try_new(schema).expect("layout")
    }

    fn padded(s: &str, len: usize) -> Vec<u8> {
        let mut v = s.as_bytes().to_vec();
        assert!(v.len() <= len);
        v.resize(len, 0);
        v
    }

    /// A ping response for the reference schema: timestamp u32, angles f32 x
    /// `channels`, encoder_value i32, encoder_button bool.
    fn ping_response(channels: u8) -> Vec<u8> {
        let mut v = PING_HEADER.to_vec();
        v.extend(padded("v1.0.0", FW_LEN));
        v.extend(padded("KER-v1.0.0", HW_LEN));
        v.extend(padded("2026-05-25", UPDATED_LEN));
        v.push(4);
        for (key, type_id, count) in [
            ("timestamp", 0u8, 1u8),
            ("angles", 5, channels),
            ("encoder_value", 3, 1),
            ("encoder_button", 6, 1),
        ] {
            v.extend(padded(key, KEY_LEN));
            v.push(type_id);
            v.push(count);
        }
        v
    }

    fn reference_schema(channels: u8) -> Schema {
        match Schema::parse_ping(&ping_response(channels)) {
            PingParse::Parsed { schema, .. } => schema,
            other => panic!("expected parse, got {other:?}"),
        }
    }

    /// A stream packet for the reference schema, with a valid checksum.
    fn stream_packet(timestamp: u32, angles: &[f32], encoder: i32, button: bool) -> Vec<u8> {
        let mut payload = timestamp.to_le_bytes().to_vec();
        for a in angles {
            payload.extend(a.to_le_bytes());
        }
        payload.extend(encoder.to_le_bytes());
        payload.push(button as u8);
        let mut packet = STREAM_HEADER.to_vec();
        packet.push(xor_checksum(&payload));
        packet.splice(2..2, payload);
        packet
    }

    #[test]
    fn ping_round_trips_metadata_and_fields() {
        let response = ping_response(16);
        let PingParse::Parsed { schema, consumed } = Schema::parse_ping(&response) else {
            panic!("expected parse");
        };
        assert_eq!(consumed, response.len());
        assert_eq!(schema.metadata.firmware, "v1.0.0");
        assert_eq!(schema.metadata.hardware, "KER-v1.0.0");
        assert_eq!(schema.metadata.updated, "2026-05-25");
        assert_eq!(schema.fields.len(), 4);
        assert_eq!(schema.fields[1].key, "angles");
        assert_eq!(schema.fields[1].ty, FieldType::F32);
        assert_eq!(schema.fields[1].count, 16);
        let layout = layout_of(&schema);
        assert_eq!(layout.payload_len(), 4 + 16 * 4 + 4 + 1);
    }

    #[test]
    fn ping_parses_mid_garbage_and_reports_consumed() {
        let mut buf = vec![0x11, 0xA5, 0x22];
        buf.extend(ping_response(8));
        let PingParse::Parsed { consumed, .. } = Schema::parse_ping(&buf) else {
            panic!("expected parse");
        };
        assert_eq!(consumed, buf.len());
    }

    #[test]
    fn ping_needs_more_on_every_truncation() {
        let response = ping_response(16);
        for len in 0..response.len() {
            assert!(
                matches!(Schema::parse_ping(&response[..len]), PingParse::NeedMore),
                "truncation at {len} must ask for more"
            );
        }
    }

    #[test]
    fn ping_rejects_an_unknown_type_id() {
        let mut response = ping_response(16);
        let angles_entry = PING_FIXED_LEN + FIELD_ENTRY_LEN;
        response[angles_entry + KEY_LEN] = 7;
        assert!(matches!(
            Schema::parse_ping(&response),
            PingParse::Invalid(ProtocolError::UnknownTypeId(7))
        ));
    }

    #[test]
    fn layout_decodes_a_packet_exactly() {
        let schema = reference_schema(3);
        let layout = layout_of(&schema);
        assert_eq!(layout.angle_count(), 3);
        assert!(layout.has_button());

        let packet = stream_packet(7, &[10.0, -20.5, 30.25], -4, true);
        let mut deframer = Deframer::new(layout.payload_len());
        deframer.push(&packet);
        let payload = deframer.next_payload().expect("one frame").expect("valid");
        assert_eq!(
            layout.parse(&payload),
            KerFrame {
                timestamp: 7,
                angles_deg: vec![10.0, -20.5, 30.25],
                encoder_value: -4,
                encoder_button: true,
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
    fn layout_defaults_the_optional_fields() {
        let schema = Schema {
            metadata: reference_schema(1).metadata,
            fields: vec![FieldDesc {
                key: "angles".into(),
                ty: FieldType::F32,
                count: 2,
            }],
        };
        let layout = layout_of(&schema);
        assert!(!layout.has_button());
        let frame = layout.parse(&[0, 0, 128, 63, 0, 0, 0, 64]);
        assert_eq!(frame.angles_deg, vec![1.0, 2.0]);
        assert_eq!(frame.timestamp, 0);
        assert_eq!(frame.encoder_value, 0);
        assert!(!frame.encoder_button);
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
        // `parse` exactly the bytes it decodes. This is the invariant that used
        // to be an assertion inside `parse`.
        let schema = Schema {
            metadata: reference_schema(1).metadata,
            fields: vec![
                FieldDesc {
                    key: "encoder_button".into(),
                    ty: FieldType::Bool,
                    count: 1,
                },
                FieldDesc {
                    key: "angles".into(),
                    ty: FieldType::F32,
                    count: 2,
                },
                FieldDesc {
                    key: "encoder_value".into(),
                    ty: FieldType::I16,
                    count: 1,
                },
                // A field this node does not consume still occupies its bytes.
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
                // Absent from this schema, so it decodes to the documented default.
                timestamp: 0,
                angles_deg: vec![1.5, -2.5],
                encoder_value: -7,
                encoder_button: true,
            }
        );
    }

    #[test]
    fn deframer_reassembles_byte_at_a_time_delivery() {
        let schema = reference_schema(2);
        let layout = layout_of(&schema);
        let mut deframer = Deframer::new(layout.payload_len());
        let packet = stream_packet(1, &[1.0, 2.0], 0, false);
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
        let mut corrupted = stream_packet(1, &[1.0, 2.0], 0, false);
        let last = corrupted.len() - 1;
        corrupted[last] ^= 0xFF;
        deframer.push(&corrupted);
        deframer.push(&stream_packet(2, &[3.0, 4.0], 0, false));
        assert_eq!(deframer.next_payload(), Some(Err(BadChecksum)));
        let payload = deframer.next_payload().expect("frame").expect("valid");
        assert_eq!(layout.parse(&payload).timestamp, 2);
    }

    #[test]
    fn deframer_skips_leading_garbage_and_bounds_its_buffer() {
        let schema = reference_schema(2);
        let layout = layout_of(&schema);
        let mut deframer = Deframer::new(layout.payload_len());
        deframer.push(&[0x00, 0xA5, 0x00, 0xFF]);
        assert!(deframer.next_payload().is_none());
        assert!(deframer.buf.is_empty(), "garbage must not accumulate");
        deframer.push(&stream_packet(3, &[0.5, -0.5], 9, true));
        let payload = deframer.next_payload().expect("frame").expect("valid");
        assert_eq!(layout.parse(&payload).encoder_value, 9);
    }

    #[test]
    fn deframer_keeps_a_trailing_possible_header_byte() {
        let schema = reference_schema(2);
        let layout = layout_of(&schema);
        let mut deframer = Deframer::new(layout.payload_len());
        let packet = stream_packet(4, &[1.0, 1.0], 0, false);
        deframer.push(&[0x33, STREAM_HEADER[0]]);
        assert!(deframer.next_payload().is_none());
        // The 0xA5 tail must survive so a packet split right after it still parses.
        deframer.push(&packet[1..]);
        assert!(deframer.next_payload().expect("frame").is_ok());
    }
}
