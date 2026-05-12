# Integration Tests with v4l2loopback

This directory contains integration tests that use virtual camera devices to test the full camera capture pipeline without requiring physical hardware.

## Prerequisites

### Install Dependencies (Ubuntu/Debian)
```bash
# Install v4l2loopback kernel module
sudo apt-get update
sudo apt-get install v4l2loopback-dkms v4l2loopback-utils

# Install ffmpeg for streaming test patterns
sudo apt-get install ffmpeg

# Verify installations
which ffmpeg
ls /dev/video*
```

### Load v4l2loopback Module
```bash
# Load the module and create a virtual device at /dev/video10
# Note: exclusive_caps=0 is required for compatibility with nokhwa
sudo modprobe v4l2loopback devices=1 video_nr=10 exclusive_caps=0 max_buffers=2 card_label="TestCamera"

# Verify the device was created
ls -l /dev/video10
v4l2-ctl --list-devices

# To make it persistent across reboots (optional)
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo "options v4l2loopback devices=1 video_nr=10 exclusive_caps=0 max_buffers=2 card_label='TestCamera'" | \
  sudo tee /etc/modprobe.d/v4l2loopback.conf
```

### Unload Module (if needed)
```bash
sudo modprobe -r v4l2loopback
```

## Running Integration Tests

**Important:** Integration tests must run single-threaded (`--test-threads=1`) because they all share the same v4l2loopback device (`/dev/video10`). Running tests in parallel will cause device conflicts and test failures.

### Run All Integration Tests
```bash
# From the uvc_camera directory
# IMPORTANT: Use --test-threads=1 to avoid device conflicts
cargo test --test integration_tests -- --ignored --test-threads=1

# With output
cargo test --test integration_tests -- --ignored --test-threads=1 --nocapture

# Run specific test
cargo test --test integration_tests test_capture_frames_from_virtual_camera -- --ignored --test-threads=1 --nocapture
```

### Run All Tests (Unit + Integration)
```bash
cargo test -- --include-ignored --test-threads=1
```

## Test Overview

### Available Integration Tests

1. **test_open_virtual_camera** - Verifies basic camera opening
2. **test_capture_frames_from_virtual_camera** - Captures and validates multiple frames
3. **test_capture_color_bars** - Tests with SMPTE color bars pattern
4. **test_parse_device_and_capture** - Tests device path parsing with real device
5. **test_frame_rate_timing** - Validates frame rate timing

### Test Helper: VirtualCamera

The `VirtualCamera` helper automatically:
- Checks if v4l2loopback device exists
- Starts ffmpeg streaming test pattern
- Cleans up ffmpeg process on drop
- Provides RAII guarantees

```rust
// Example usage
let vcam = VirtualCamera::new(10, 640, 480, 30)?;
// Use /dev/video10 for testing
// vcam automatically stops streaming when dropped
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration-test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y v4l2loopback-dkms ffmpeg
      
      - name: Load v4l2loopback
        run: |
          sudo modprobe v4l2loopback devices=1 video_nr=10 exclusive_caps=0 max_buffers=2 card_label="TestCamera"
          v4l2-ctl --list-devices
      
      - name: Run integration tests
        run: cargo test -- --include-ignored --test-threads=1
```

## Troubleshooting

### "Device not found" Error
- Check if v4l2loopback is loaded: `lsmod | grep v4l2loopback`
- Check if device exists: `ls -l /dev/video10`
- Try loading manually: `sudo modprobe v4l2loopback devices=1 video_nr=10`

### "ffmpeg not found" Error
- Install ffmpeg: `sudo apt-get install ffmpeg`
- Verify installation: `which ffmpeg`

### Permission Denied
```bash
# Add your user to the video group
sudo usermod -a -G video $USER

# Log out and back in for changes to take effect
```

### Module Won't Load
```bash
# Check kernel compatibility
uname -r
apt-cache policy v4l2loopback-dkms

# Rebuild module if needed
sudo dkms remove v4l2loopback/0.12.7 --all
sudo apt-get install --reinstall v4l2loopback-dkms
```

### Device Already in Use
```bash
# Find processes using the device
sudo lsof /dev/video10

# Kill stale processes
pkill ffmpeg
```

## Manual Testing with Virtual Camera

You can manually test with the virtual device:

```bash
# Start streaming test pattern
ffmpeg -re -f lavfi -i testsrc=size=640x480:rate=30 \
  -pix_fmt rgb24 -f v4l2 /dev/video10

# In another terminal, view the stream
ffplay /dev/video10

# Or capture with your node
cargo run
```

## Alternative: Docker Testing

If you prefer containerized testing:

```dockerfile
FROM rust:latest

RUN apt-get update && apt-get install -y \
    linux-headers-generic \
    v4l2loopback-dkms \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

CMD ["bash", "-c", "modprobe v4l2loopback && cargo test -- --include-ignored --test-threads=1"]
```

Note: Docker approach requires `--privileged` flag for module loading.

## References

- [v4l2loopback Documentation](https://github.com/umlaeute/v4l2loopback)
- [FFmpeg Virtual Devices](https://trac.ffmpeg.org/wiki/Capture/Webcam#Linux)
- [nokhwa Camera Library](https://docs.rs/nokhwa/)
