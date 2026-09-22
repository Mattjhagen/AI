#!/bin/bash
# Deploy trained DeepSeek-R1 checkpoint to AWS EC2 production
# Run this after training is complete and you have a checkpoint

set -e

# Configuration
CHECKPOINT_DIR="${1:-./shaggoth-trained}"
S3_BUCKET="s3://shaggoth-models"
S3_KEY="deepseek-r1-$(date +%Y%m%d)"
EC2_INSTANCE="${EC2_INSTANCE:-aws}"  # SSH alias or IP
EC2_MODEL_DIR="/opt/shaggoth/models"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

error() {
    echo -e "${RED}❌ Error: $1${NC}" >&2
    exit 1
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

echo "🚀 DeepSeek-R1 Deployment to EC2"
echo "=================================="
echo ""

# Validate checkpoint
if [ ! -d "$CHECKPOINT_DIR" ]; then
    error "Checkpoint directory not found: $CHECKPOINT_DIR"
fi

if [ ! -f "$CHECKPOINT_DIR/adapter_config.json" ]; then
    error "Invalid checkpoint: adapter_config.json not found"
fi

info "Checkpoint: $CHECKPOINT_DIR"
info "Size: $(du -sh "$CHECKPOINT_DIR" | cut -f1)"
echo ""

# Step 1: Upload to S3
echo "📤 Step 1: Uploading checkpoint to S3..."
echo "Destination: $S3_BUCKET/$S3_KEY/"
echo ""

read -p "Upload to S3? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    aws s3 sync "$CHECKPOINT_DIR" "$S3_BUCKET/$S3_KEY/" \
        --exclude ".git/*" \
        --exclude "*.log" \
        --exclude "__pycache__/*"

    success "Uploaded to S3"

    # Create a "latest" symlink
    aws s3 cp "$S3_BUCKET/$S3_KEY/" "$S3_BUCKET/deepseek-r1-latest/" --recursive

    success "Created latest version pointer"
else
    warn "Skipping S3 upload"
fi

echo ""

# Step 2: Deploy to EC2
echo "📦 Step 2: Deploying to EC2 instance..."
echo "Instance: $EC2_INSTANCE"
echo ""

read -p "Deploy to EC2? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then

    info "Connecting to EC2..."

    # Create remote directory
    ssh "$EC2_INSTANCE" "sudo mkdir -p $EC2_MODEL_DIR && sudo chown \$(whoami) $EC2_MODEL_DIR"

    # Sync from S3 to EC2
    info "Downloading checkpoint from S3 to EC2..."
    ssh "$EC2_INSTANCE" "aws s3 sync $S3_BUCKET/$S3_KEY/ $EC2_MODEL_DIR/deepseek-r1/"

    success "Checkpoint deployed to EC2:$EC2_MODEL_DIR/deepseek-r1/"

else
    warn "Skipping EC2 deployment"
    exit 0
fi

echo ""

# Step 3: Update configuration
echo "⚙️  Step 3: Updating Shaggoth configuration..."
echo ""

read -p "Update Shaggoth config? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then

    # Update environment variable
    ssh "$EC2_INSTANCE" "echo 'export DEEPSEEK_CHECKPOINT=$EC2_MODEL_DIR/deepseek-r1' | sudo tee -a /etc/environment"

    # Update systemd service if exists
    if ssh "$EC2_INSTANCE" "sudo test -f /etc/systemd/system/shaggoth.service"; then
        info "Updating systemd service..."

        ssh "$EC2_INSTANCE" << 'EOF'
# Add environment variable to service
sudo sed -i '/\[Service\]/a Environment="DEEPSEEK_CHECKPOINT=/opt/shaggoth/models/deepseek-r1"' \
    /etc/systemd/system/shaggoth.service

# Reload daemon
sudo systemctl daemon-reload
EOF

        success "Systemd service updated"
    fi

    success "Configuration updated"

else
    warn "Skipping configuration update"
    warn "Remember to manually set: export DEEPSEEK_CHECKPOINT=$EC2_MODEL_DIR/deepseek-r1"
fi

echo ""

# Step 4: Restart service
echo "🔄 Step 4: Restarting Shaggoth service..."
echo ""

read -p "Restart Shaggoth service? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then

    if ssh "$EC2_INSTANCE" "sudo test -f /etc/systemd/system/shaggoth.service"; then
        info "Restarting shaggoth.service..."
        ssh "$EC2_INSTANCE" "sudo systemctl restart shaggoth"
        sleep 3
        ssh "$EC2_INSTANCE" "sudo systemctl status shaggoth --no-pager"

        success "Service restarted"
    else
        warn "Systemd service not found"
        info "Start manually: python3 -m shaggoth serve --port 8420"
    fi

else
    warn "Skipping service restart"
fi

echo ""

# Step 5: Verify deployment
echo "✅ Step 5: Verifying deployment..."
echo ""

info "Testing DeepSeek endpoint..."

# Get EC2 public IP
EC2_IP=$(ssh "$EC2_INSTANCE" "curl -s http://169.254.169.254/latest/meta-data/public-ipv4" 2>/dev/null || echo "localhost")

# Test status endpoint
if ssh "$EC2_INSTANCE" "curl -s http://localhost:8420/deepseek/status" > /tmp/deepseek-status.json 2>/dev/null; then
    cat /tmp/deepseek-status.json | python3 -m json.tool 2>/dev/null || cat /tmp/deepseek-status.json
    success "DeepSeek endpoint responding"
else
    error "DeepSeek endpoint not responding"
fi

echo ""
echo "=================================="
echo "🎉 DEPLOYMENT COMPLETE"
echo "=================================="
echo ""
echo "📊 Deployment Summary:"
echo "  S3 Backup: $S3_BUCKET/$S3_KEY/"
echo "  EC2 Location: $EC2_INSTANCE:$EC2_MODEL_DIR/deepseek-r1/"
echo "  Endpoint: http://$EC2_IP:8420/deepseek/status"
echo ""
echo "🧪 Test Commands:"
echo ""
echo "# Check status"
echo "curl http://$EC2_IP:8420/deepseek/status | jq"
echo ""
echo "# Test chat"
echo "curl -X POST http://$EC2_IP:8420/deepseek/chat \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"message\": \"Explain machine learning\"}' | jq"
echo ""
echo "# Test reasoning"
echo "curl -X POST http://$EC2_IP:8420/deepseek/reason \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"query\": \"Compare supervised and unsupervised learning\"}' | jq"
echo ""
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Test endpoints from your mobile apps"
echo "  2. Monitor inference latency"
echo "  3. Update mobile apps to use /deepseek/reason for complex queries"
echo "  4. Set up CloudWatch monitoring"
echo ""
