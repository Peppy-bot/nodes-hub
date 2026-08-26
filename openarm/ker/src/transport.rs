// Blocking byte transports to the KER's M5Stack CoreS3: USB vendor mode
// (rusb) or serial CDC (serialport). Both poll reads with a short timeout so
// the reader thread can check cancellation between reads; a timeout surfaces
// as `Ok(0)`, device loss as `Err`.

use std::io;
use std::time::{Duration, Instant};

/// A launcher `transport` this node has no link for.
#[derive(Debug, thiserror::Error)]
#[error("transport must be 'usb' or 'serial', got '{0}'")]
pub struct UnknownTransport(pub String);

/// Espressif's vendor id; the CoreS3's native USB.
const USB_VID: u16 = 0x303A;
/// The KER firmware's vendor-mode product id.
const USB_PID: u16 = 0x4002;

const READ_TIMEOUT: Duration = Duration::from_millis(100);
const WRITE_TIMEOUT: Duration = Duration::from_millis(200);
/// Bound on draining stale bytes at connect, mirroring the reference impl.
const FLUSH_WINDOW: Duration = Duration::from_millis(200);
const FLUSH_READ_TIMEOUT: Duration = Duration::from_millis(10);

/// Which link to the device to open, parsed once from the node parameters.
#[derive(Debug, Clone)]
pub enum TransportConfig {
    Usb,
    Serial { port: String, baud: u32 },
}

impl TransportConfig {
    pub fn parse(transport: &str, port: &str, baud: u32) -> Result<Self, UnknownTransport> {
        match transport {
            "usb" => Ok(Self::Usb),
            "serial" => Ok(Self::Serial {
                port: port.to_string(),
                baud,
            }),
            other => Err(UnknownTransport(other.to_string())),
        }
    }
}

pub trait KerTransport: Send {
    fn write_all(&mut self, bytes: &[u8]) -> io::Result<()>;
    /// Read available bytes into `buf`; `Ok(0)` means nothing arrived in time.
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize>;
    /// Drop whatever the device sent before we were listening.
    fn flush_input(&mut self) -> io::Result<()>;
}

pub fn open(cfg: &TransportConfig) -> io::Result<Box<dyn KerTransport>> {
    match cfg {
        TransportConfig::Usb => Ok(Box::new(UsbTransport::open()?)),
        TransportConfig::Serial { port, baud } => Ok(Box::new(SerialTransport::open(port, *baud)?)),
    }
}

struct UsbTransport {
    handle: rusb::DeviceHandle<rusb::GlobalContext>,
    ep_in: u8,
    ep_out: u8,
}

impl UsbTransport {
    fn open() -> io::Result<Self> {
        let handle = rusb::open_device_with_vid_pid(USB_VID, USB_PID).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("USB device {USB_VID:04x}:{USB_PID:04x} not found (KER unplugged, or in serial mode?)"),
            )
        })?;
        // Endpoints are discovered, not hardcoded: take the first interface
        // exposing a bulk pair, which is the vendor-mode data interface.
        let config = handle
            .device()
            .active_config_descriptor()
            .map_err(io::Error::other)?;
        let (iface, ep_in, ep_out) = config
            .interfaces()
            .flat_map(|i| i.descriptors())
            .find_map(|desc| {
                let bulk = |dir| {
                    desc.endpoint_descriptors()
                        .find(|e| {
                            e.transfer_type() == rusb::TransferType::Bulk && e.direction() == dir
                        })
                        .map(|e| e.address())
                };
                Some((
                    desc.interface_number(),
                    bulk(rusb::Direction::In)?,
                    bulk(rusb::Direction::Out)?,
                ))
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "no bulk in/out interface on the device",
                )
            })?;
        // Linux may have bound a kernel driver; detach it for the claim.
        let _ = handle.set_auto_detach_kernel_driver(true);
        handle.claim_interface(iface).map_err(io::Error::other)?;
        Ok(Self {
            handle,
            ep_in,
            ep_out,
        })
    }
}

impl KerTransport for UsbTransport {
    fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
        let mut sent = 0;
        while sent < bytes.len() {
            sent += self
                .handle
                .write_bulk(self.ep_out, &bytes[sent..], WRITE_TIMEOUT)
                .map_err(io::Error::other)?;
        }
        Ok(())
    }

    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self.handle.read_bulk(self.ep_in, buf, READ_TIMEOUT) {
            Ok(n) => Ok(n),
            Err(rusb::Error::Timeout) => Ok(0),
            Err(e) => Err(io::Error::other(e)),
        }
    }

    fn flush_input(&mut self) -> io::Result<()> {
        let mut sink = [0u8; 512];
        let deadline = Instant::now() + FLUSH_WINDOW;
        while Instant::now() < deadline {
            match self
                .handle
                .read_bulk(self.ep_in, &mut sink, FLUSH_READ_TIMEOUT)
            {
                Ok(_) => continue,
                Err(rusb::Error::Timeout) => break,
                Err(e) => return Err(io::Error::other(e)),
            }
        }
        Ok(())
    }
}

struct SerialTransport {
    port: Box<dyn serialport::SerialPort>,
}

impl SerialTransport {
    fn open(port: &str, baud: u32) -> io::Result<Self> {
        let port = serialport::new(port, baud)
            .timeout(READ_TIMEOUT)
            .open()
            .map_err(io::Error::other)?;
        Ok(Self { port })
    }
}

impl KerTransport for SerialTransport {
    fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.port.write_all(bytes)
    }

    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self.port.read(buf) {
            Ok(n) => Ok(n),
            Err(e) if e.kind() == io::ErrorKind::TimedOut => Ok(0),
            Err(e) => Err(e),
        }
    }

    fn flush_input(&mut self) -> io::Result<()> {
        self.port
            .clear(serialport::ClearBuffer::Input)
            .map_err(io::Error::other)
    }
}
