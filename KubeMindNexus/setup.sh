#!/bin/bash
# Setup script for KubeMindNexus development environment

# Create virtual environment using uv
echo "Creating Python virtual environment with uv..."
uv venv

# Activate virtual environment
if [ -d ".venv/bin" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Activating virtual environment (Windows)..."
    source .venv/Scripts/activate
fi

# Install dependencies
echo "Installing project dependencies..."
uv pip install -e .

# Create default configuration directory and copy initial configuration
echo "Setting up configuration..."
CONFIG_DIR="${HOME}/.kubemindnexus"
mkdir -p "${CONFIG_DIR}"

if [ ! -f "${CONFIG_DIR}/config.json" ]; then
    echo "Creating default configuration file..."
    cp config.json "${CONFIG_DIR}/config.json"
    echo "Default configuration created at ${CONFIG_DIR}/config.json"
else
    echo "Configuration file already exists at ${CONFIG_DIR}/config.json"
    echo "If you want to use the new configuration format, backup your existing config and copy the new one:"
    echo "cp ${CONFIG_DIR}/config.json ${CONFIG_DIR}/config.json.backup"
    echo "cp config.json ${CONFIG_DIR}/config.json"
fi

echo "Installation complete!"
echo "You can activate the virtual environment with 'source .venv/bin/activate' (Linux/Mac) or '.venv\\Scripts\\activate' (Windows)"
echo "Run 'python -m kubemindnexus' to start the application"
