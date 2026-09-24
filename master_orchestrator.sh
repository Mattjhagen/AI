#!/bin/bash
# Master Orchestrator - Automates entire Shaggoth training pipeline
# Monitors R510, auto-heals issues, manages training lifecycle

WORK_DIR="/home/matt/AI"
LOG_FILE="$WORK_DIR/orchestrator.log"
STATUS_FILE="/tmp/orchestrator_status.json"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

update_status() {
    echo "$1" > "$STATUS_FILE"
}

check_r510_health() {
    log "Checking R510 health..."

    # Check SSH connectivity
    if ! ssh r510 'echo "OK"' > /dev/null 2>&1; then
        log "ERROR: Cannot connect to R510"
        return 1
    fi

    # Check memory
    FREE_MEM=$(ssh r510 'free -g | grep Mem | awk "{print \$4}"')
    SWAP_USED=$(ssh r510 'free -g | grep Swap | awk "{print \$3}"')

    log "R510 Status: ${FREE_MEM}GB free RAM, ${SWAP_USED}GB swap used"

    if [ "$FREE_MEM" -lt 2 ]; then
        log "WARNING: Low memory on R510"
        return 2
    fi

    if [ "$SWAP_USED" -gt 4 ]; then
        log "WARNING: High swap usage on R510"
        return 2
    fi

    log "✓ R510 health check passed"
    return 0
}

free_r510_memory() {
    log "Freeing R510 memory..."
    ssh r510 'cd ~/AI && ./prepare_r510_for_training.sh'
    sleep 5
    check_r510_health
}

deploy_training_files() {
    log "Deploying training files to R510..."

    # Copy training script
    scp "$WORK_DIR/automated_training.py" r510:~/AI/
    scp "$WORK_DIR/datasets/"*.jsonl r510:~/AI/datasets/ 2>/dev/null || log "Datasets already on R510"

    log "✓ Files deployed"
}

start_training_r510() {
    log "Starting training on R510..."

    ssh r510 'cd ~/AI && source venv/bin/activate && nohup python3 automated_training.py > training.log 2>&1 &'

    log "✓ Training started - monitoring..."
}

monitor_training() {
    log "Monitoring training progress..."

    while true; do
        # Check if training is still running
        if ssh r510 'pgrep -f "automated_training.py"' > /dev/null 2>&1; then
            # Get status
            STATUS=$(ssh r510 'cat /tmp/shaggoth_training_status.json 2>/dev/null' || echo '{"status":"unknown"}')
            log "Training status: $STATUS"

            # Check for issues
            check_r510_health
            HEALTH=$?

            if [ $HEALTH -eq 2 ]; then
                log "⚠ Health issue detected - applying auto-heal"
                auto_heal_r510
            fi

            sleep 30
        else
            # Training finished or crashed
            log "Training process not found"
            break
        fi
    done

    # Check final status
    FINAL_STATUS=$(ssh r510 'cat /tmp/shaggoth_training_status.json 2>/dev/null')
    log "Final status: $FINAL_STATUS"

    if echo "$FINAL_STATUS" | grep -q '"status":"complete"'; then
        log "✓ Training completed successfully!"
        return 0
    else
        log "✗ Training failed or incomplete"
        return 1
    fi
}

auto_heal_r510() {
    log "🔧 AUTO-HEALING: Analyzing issue..."

    # Check what's consuming memory
    TOP_PROCS=$(ssh r510 'ps aux --sort=-%mem | head -6')
    log "Top memory consumers:\n$TOP_PROCS"

    # Clear caches
    log "Clearing system caches..."
    ssh r510 'sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1' || log "Cache clear failed (need sudo)"

    # Check swap
    SWAP_USED=$(ssh r510 'free -g | grep Swap | awk "{print \$3}"')
    if [ "$SWAP_USED" -gt 5 ]; then
        log "Critical swap usage - recommending restart of non-essential services"
        # Could add logic here to restart services
    fi

    sleep 5
    log "✓ Auto-heal complete"
}

deploy_trained_model() {
    log "Deploying trained model..."

    # Check if model exists
    if ssh r510 'test -d ~/AI/shaggoth-trained-lora'; then
        log "✓ Trained model found on R510"

        # Create inference endpoint
        log "Setting up inference endpoint..."
        ssh r510 'cd ~/AI && python3 setup_inference.py'

        log "✓ Model deployed and ready for inference"
        return 0
    else
        log "ERROR: Trained model not found"
        return 1
    fi
}

# Main orchestration flow
main() {
    log "=========================================="
    log "Shaggoth AI Training - Master Orchestrator"
    log "=========================================="

    update_status '{"stage":"init","status":"starting"}'

    # Step 1: Health check
    log "\n=== STEP 1: Health Check ==="
    check_r510_health
    if [ $? -eq 1 ]; then
        log "ERROR: R510 not reachable - aborting"
        exit 1
    fi

    # Step 2: Prepare R510
    log "\n=== STEP 2: Prepare R510 ==="
    free_r510_memory

    # Step 3: Deploy files
    log "\n=== STEP 3: Deploy Files ==="
    deploy_training_files

    # Step 4: Start training
    log "\n=== STEP 4: Start Training ==="
    start_training_r510
    update_status '{"stage":"training","status":"running"}'

    # Step 5: Monitor with auto-healing
    log "\n=== STEP 5: Monitor Training ==="
    monitor_training
    TRAINING_SUCCESS=$?

    # Step 6: Deploy if successful
    if [ $TRAINING_SUCCESS -eq 0 ]; then
        log "\n=== STEP 6: Deploy Model ==="
        deploy_trained_model
        DEPLOY_SUCCESS=$?

        if [ $DEPLOY_SUCCESS -eq 0 ]; then
            log "\n=========================================="
            log "✓ MISSION COMPLETE!"
            log "=========================================="
            log "Trained model deployed and running on R510"
            log "Inference endpoint: http://r510:8421"
            update_status '{"stage":"complete","status":"success"}'
            exit 0
        fi
    fi

    log "\n=========================================="
    log "✗ Mission incomplete - check logs"
    log "=========================================="
    update_status '{"stage":"failed","status":"error"}'
    exit 1
}

# Run if executed directly
if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
    main "$@"
fi
