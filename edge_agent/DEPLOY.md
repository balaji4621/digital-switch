# Cross-Compilation and Deployment for TEJAS Edge Agent

## Overview

This document describes the process of cross-compiling the Rust-based TEJAS edge agent (ARM64) for embedded deployment on target boards (Raspberry Pi, BeagleBone, NVIDIA Jetson, etc.). The project targets `aarch64-unknown-linux-gnu` to produce a self-contained binary that can run on Linux-based embedded systems.

## Prerequisites

### Host Environment (x86_64 Linux or macOS)

- Rust toolchain with `cargo` and `rustup`
- Cross-compiler toolchain for `aarch64-unknown-linux-gnu`
  - Option 1: Install `cross` tool (recommended)
  - Option 2: Install `aarch64-unknown-linux-gnu-gcc` system package

### Target Board

The target board should have:
- ARMv8-A architecture or compatible
- Filesystem access to mount the binary and assets
- Optional: GPIO pins accessible via `/sys/class/gpio` (for relay control)

## Installation Steps

### 1. Install Cross-Compilation Tools

#### Option 1: Using the `cross` tool (recommended)

```bash
# Install cross
cargo install cross

# Verify installation
cross --version
```

#### Option 2: Using system package manager

**Ubuntu/Debian:**
```bash
apt update && apt install gcc-aarch64-linux-gnu gcc-aarch64-linux-gnu libc6-dev-aarch64

# Set up cross-compilation environment
cat >> ~/.cargo/config <<'EOF'
[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"
EOF
```

**Fedora/RHEL:**
```bash
sudo dnf install gcc-aarch64-linux-gnu glibc-devel.aarch64

# Set up cross-compilation environment
cat >> ~/.cargo/config <<'EOF'
[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"
EOF
```

**Arch Linux:**
```bash
sudo pacman -S aarch64-linux-gnu-gcc

# Set up cross-compilation environment
cat >> ~/.cargo/config <<'EOF'
[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"
EOF
```

### 2. Build the Release Binary

```bash
cross build --target aarch64-unknown-linux-gnu --release
```

This produces:
```
target/aarch64-unknown-linux-gnu/release/edge_agent
```

### 3. Prepare Deployment Directory on Target Board

Connect to the target board via SSH and create the deployment directory:

```bash
sudo mkdir -p /opt/tejas/super_edge_engine
sudo chmod 755 /opt/tejas
```

### 4. Copy Files to Target Board

Copy the following files from the host to the target board:

```bash
# Copy the compiled binary
cp target/aarch64-unknown-linux-gnu/release/edge_agent /opt/tejas/
sudo chmod +x /opt/tejas/edge_agent

# Copy the ONNX model and scalers
cp ../super_edge_engine/laya_student_engine.onnx /opt/tejas/
cp ../super_edge_engine/scaler_mean.npy /opt/tejas/
cp ../super_edge_engine/scaler_scale.npy /opt/tejas/

# Set appropriate permissions
sudo chown root:root /opt/tejas/edge_agent
sudo chmod 755 /opt/tejas/edge_agent
sudo chown root:root /opt/tejas/laya_student_engine.onnx
sudo chown root:root /opt/tejas/scaler_mean.npy
sudo chown root:root /opt/tejas/scaler_scale.npy
```

### 5. Verify Installation

On the target board:

```bash
# Check if the binary exists
ls -la /opt/tejas/

# Test running the binary (should output help/version info)
/opt/tejas/edge_agent
```

## Alternative: Static Linking (for Reduced Dependencies)

For more portable deployment, consider static linking. However, note that ARM64 static binaries are larger and may require additional system libraries.

### Static Build (with musl)

If using musl target (if available):

```bash
cross build --target aarch64-unknown-linux-musl --release
```

## Configuration and Assets

### Required Files

The deployment requires three files:

1. **`edge_agent`** — The compiled binary (ARM64 executable)
2. **`laya_student_engine.onnx`** — ONNX model file (12,809 bytes)
3. **`scaler_mean.npy`** — Telemetry offset array (160 bytes)
4. **`scaler_scale.npy`** — Telemetry scale divisor array (160 bytes)

### File Locations

**Host (for build):**
- Binary: `target/aarch64-unknown-linux-gnu/release/edge_agent`
- Model: `../super_edge_engine/laya_student_engine.onnx`
- Scalers: `../super_edge_engine/scaler_mean.npy`, `../super_edge_engine/scaler_scale.npy`

**Target Board:**
- All files go into `/opt/tejas/` directory

### Configuration Environment Variables

The edge agent can accept optional environment variables:

```bash
export TEJAS_LOG_LEVEL=info  # Log level (debug, info, warn, error)
export TEJAS_HAZARD_THRESHOLD=0.85  # Threshold for trip hazard detection
export TEJAS_HARDWARE_SIMULATION=0  # 0 = real hardware, 1 = simulation (for testing)
```

### Testing on Target Board

```bash
# Navigate to deployment directory
cd /opt/tejas/

# Run the edge agent (will serve via stdio, typically handled by systemd or supervisor)
./edge_agent &

# Check logs if running in background
tail -f /var/log/tejas_edge_agent.log
```

For MCP (Model Context Protocol) interaction, the binary listens on stdio. This is typically managed by an MCP client/server bridge or systemd with appropriate redirection.

## Troubleshooting

### Common Issues

#### Permission Denied
```bash
# On target board
sudo chown root:root /opt/tejas/edge_agent
sudo chmod 755 /opt/tejas/edge_agent
```

#### Missing Dependencies
The binary is statically linked with musl (if built with `--target aarch64-unknown-linux-musl`) and should not have external library dependencies.

If runtime errors occur:
- Verify the target board architecture matches `aarch64-unknown-linux-gnu`
- Check filesystem permissions and ownership
- Ensure sufficient disk space

#### MCP Connection Issues
The binary uses stdio for MCP communication. For direct interaction:
- Use MCP client tools that support stdio
- Or implement a simple pipe-based wrapper
- For systemd: use `StandardIO=inherit` or `StandardInput=tty`

## Deployment Scripts

### Bash Deployment Script (`deploy.sh`)

```bash
#!/bin/bash

set -euo pipefail

TARGET_HOST="$1"
TARGET_USER="$2"
TARGET_DIR="/opt/tejas"

# SSH command template
SSH_CMD="ssh $TARGET_USER@$TARGET_HOST"

# Build on host
./cross build --target aarch64-unknown-linux-gnu --release

# Create deployment directory
ssh $TARGET_HOST "mkdir -p $TARGET_DIR"

# Copy files
ssh $TARGET_HOST "scp target/aarch64-unknown-linux-gnu/release/edge_agent $TARGET_USER@$TARGET_HOST:$TARGET_DIR/"
ssh $TARGET_HOST "scp ../super_edge_engine/laya_student_engine.onnx $TARGET_USER@$TARGET_HOST:$TARGET_DIR/"
ssh $TARGET_HOST "scp ../super_edge_engine/scaler_mean.npy $TARGET_USER@$TARGET_HOST:$TARGET_DIR/"
ssh $TARGET_HOST "scp ../super_edge_engine/scaler_scale.npy $TARGET_USER@$TARGET_HOST:$TARGET_DIR/"

# Set permissions
ssh $TARGET_HOST "chown root:root $TARGET_DIR/edge_agent && chmod 755 $TARGET_DIR/edge_agent"

# Verify installation
ssh $TARGET_HOST "$TARGET_DIR/edge_agent"

echo "Deployment completed successfully!"
```

### systemd Service Unit (`tejas_edge_agent.service`)

```ini
[Unit]
Description=TEJAS Edge AI Agent
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/tejas
ExecStart=/opt/tejas/edge_agent
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Installation via Package Manager (Debian/Ubuntu)

Create `/etc/apt/sources.list.d/tejas.list`:
```
deb http://localhost:8080/debian stable main
```

Add signing key and install:
```bash
apt update
apt install tejas-edge-agent
```

## Advanced Features

### Multiple Hardware Interfaces
The hardware abstraction layer supports different GPIO chip implementations:
- **Raspberry Pi** (`/sys/class/gpio`)
- **BeagleBone** (`/sys/class/gioplite`)
- **NVIDIA Jetson** (custom sysfs paths)

### Hardware Simulation for Development
Set `TEJAS_HARDWARE_SIMULATION=1` to simulate hardware actions during development:
```bash
export TEJAS_HARDWARE_SIMULATION=1
cargo run
```

This will output simulated hardware actions without actually accessing GPIO.

### Logging Configuration

Configure logging levels via environment variable:

```bash
export TEJAS_LOG_LEVEL=debug  # Logs include low-level hardware operations
export TEJAS_LOG_LEVEL=info   # Logs include action start/end
export TEJAS_LOG_LEVEL=warn  # Only warnings and errors
```

## License

This project is licensed under Apache 2.0.

## Author

TEJAS 2026

## Version

- Rust: 1.88+
- ORT: 2.0.0-rc.13
- rmcp: 3.5.0
- Target: aarch64-unknown-linux-gnu

## Acknowledgments

- Thanks to the Rust community for excellent tooling
- Thanks to the Laya team for the decision engine
- Special thanks to all contributors who tested and validated this deployment process