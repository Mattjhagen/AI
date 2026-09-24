#!/bin/bash
# Restore R510 services after AI training

echo "====================================="
echo "Restoring R510 Services"
echo "====================================="
echo ""

# Start Elasticsearch
echo "[1/3] Starting Elasticsearch..."
# Add your elasticsearch start command here if needed
echo "  - Manual restart may be needed"

# Start Kibana
echo "[2/3] Starting Kibana..."
# Add your kibana start command here if needed
echo "  - Manual restart may be needed"

# Start OpenCode and tunnel
echo "[3/3] Starting OpenCode..."
cd ~
nohup opencode serve --hostname 100.103.3.35 --port 4096 --print-logs --log-level INFO > /dev/null 2>&1 &
echo "  ✓ Started OpenCode"

nohup cloudflared tunnel --config .cloudflared-opencode.yml run > /dev/null 2>&1 &
echo "  ✓ Started OpenCode tunnel"

echo ""
echo "✓ Services restored!"
