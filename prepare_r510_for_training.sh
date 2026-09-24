#!/bin/bash
# Prepare R510 for AI Training
# Keeps P3 Lending (port 5173) and Shaggoth (port 8420) running

echo "=============================================="
echo "Preparing R510 for AI Training"
echo "=============================================="
echo ""
echo "This will free ~4.5GB RAM by stopping:"
echo "  - Dashboard (2.5GB) - Will move to R410"
echo "  - Elasticsearch (1.6GB)"
echo "  - Kibana (315MB)"
echo "  - OpenCode (194MB)"
echo ""
echo "Will KEEP running:"
echo "  ✓ P3 Lending (p3lending.space → port 5173)"
echo "  ✓ Shaggoth AI (port 8420)"
echo "  ✓ Cloudflare tunnel for P3 Lending"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Current memory usage:"
free -h
echo ""

# Stop Dashboard
echo "[1/5] Stopping P3-NOC-Web dashboard..."
DASHBOARD_PID=$(pgrep -f "dashboard.py --wallboard")
if [ ! -z "$DASHBOARD_PID" ]; then
    kill $DASHBOARD_PID
    echo "  ✓ Stopped dashboard (PID: $DASHBOARD_PID)"
    echo "  → Will move to R410"
else
    echo "  - Dashboard not running"
fi

# Stop Elasticsearch
echo "[2/5] Stopping Elasticsearch..."
ELASTIC_PIDS=$(pgrep -f "elasticsearch")
if [ ! -z "$ELASTIC_PIDS" ]; then
    echo "$ELASTIC_PIDS" | xargs kill
    echo "  ✓ Stopped Elasticsearch"
else
    echo "  - Elasticsearch not running"
fi

# Stop Kibana
echo "[3/5] Stopping Kibana..."
KIBANA_PID=$(pgrep -f "kibana")
if [ ! -z "$KIBANA_PID" ]; then
    kill $KIBANA_PID
    echo "  ✓ Stopped Kibana"
else
    echo "  - Kibana not running"
fi

# Stop OpenCode
echo "[4/5] Stopping OpenCode..."
OPENCODE_PID=$(pgrep -f "opencode serve")
if [ ! -z "$OPENCODE_PID" ]; then
    kill $OPENCODE_PID
    echo "  ✓ Stopped OpenCode"
else
    echo "  - OpenCode not running"
fi

# Stop OpenCode tunnel
echo "[5/5] Stopping OpenCode cloudflare tunnel..."
OPENCODE_TUNNEL=$(pgrep -f "cloudflared-opencode")
if [ ! -z "$OPENCODE_TUNNEL" ]; then
    kill $OPENCODE_TUNNEL
    echo "  ✓ Stopped OpenCode tunnel"
else
    echo "  - OpenCode tunnel not running"
fi

echo ""
echo "Waiting 10 seconds for services to release memory..."
sleep 10

echo ""
echo "New memory usage:"
free -h
echo ""
echo "✓ R510 ready for AI training!"
echo ""
echo "Services still running:"
echo "  ✓ P3 Lending on p3lending.space"
echo "  ✓ Shaggoth AI on port 8420"
echo ""
echo "To restore services: ./restore_r510_services.sh"
