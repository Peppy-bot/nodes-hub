#!/bin/bash
# Setup script for v4l2loopback integration testing

set -e

echo "Setting up v4l2loopback for UVC camera integration tests..."
echo

# Check if running as root for module loading
if [ "$EUID" -ne 0 ]; then 
    echo "This script needs sudo access to load kernel modules."
    echo "You will be prompted for your password."
    echo
fi

# Install dependencies
echo "1. Installing dependencies..."
sudo apt-get update
sudo apt-get install -y v4l2loopback-dkms v4l2loopback-utils ffmpeg

# Load v4l2loopback module
echo
echo "2. Loading v4l2loopback module..."
if lsmod | grep -q v4l2loopback; then
    echo "   v4l2loopback already loaded, reloading..."
    sudo modprobe -r v4l2loopback
fi

sudo modprobe v4l2loopback devices=1 video_nr=10 exclusive_caps=0 max_buffers=2 card_label="TestCamera"

# Verify device creation
echo
echo "3. Verifying virtual device..."
if [ -e /dev/video10 ]; then
    echo "   ✓ /dev/video10 created successfully"
    ls -l /dev/video10
else
    echo "   ✗ Failed to create /dev/video10"
    exit 1
fi

# Check permissions
echo
echo "4. Checking permissions..."
if groups | grep -q video; then
    echo "   ✓ User is in video group"
else
    echo "   ⚠ User is not in video group"
    echo "   Adding user to video group..."
    sudo usermod -a -G video $USER
    echo "   ⚠ You need to log out and back in for group changes to take effect"
fi

# Make persistent (optional)
echo
read -p "5. Make v4l2loopback persistent across reboots? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Creating module configuration..."
    echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf
    echo "options v4l2loopback devices=1 video_nr=10 exclusive_caps=0 max_buffers=2 card_label='TestCamera'" | \
        sudo tee /etc/modprobe.d/v4l2loopback.conf
    echo "   ✓ v4l2loopback will load automatically on boot"
fi

echo
echo "Setup complete! You can now run integration tests:"
echo "  cargo test -- --include-ignored --test-threads=1"
echo
echo "Note: Use --test-threads=1 to avoid device conflicts between tests"
echo
echo "To test manually:"
echo "  # Start test pattern:"
echo "  ffmpeg -re -f lavfi -i testsrc=size=640x480:rate=30 -pix_fmt rgb24 -f v4l2 /dev/video10"
echo
echo "  # View stream:"
echo "  ffplay /dev/video10"
echo
