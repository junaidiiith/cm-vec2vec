#!/bin/bash

echo "=== Process Management Script ==="

# Function to stop current process
stop_current_process() {
    echo "Stopping current process with SIGTSTP..."
    kill -TSTP $$
    sleep 1
}

# Function to find and kill test_cm_vec2vec.py processes
kill_test_processes() {
    echo "Searching for test_cm_vec2vec.py processes..."
    
    # Get all PIDs for test_cm_vec2vec.py processes
    PIDS=$(ps -aux | grep "test_cm_vec2vec.py" | grep -v grep | awk '{print $2}')
    
    if [ -z "$PIDS" ]; then
        echo "No existing test_cm_vec2vec.py processes found"
        return 0
    fi
    
    echo "Found processes: $PIDS"
    
    # Kill each process
    for pid in $PIDS; do
        echo "Killing process $pid..."
        kill -9 $pid 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "Successfully killed process $pid"
        else
            echo "Failed to kill process $pid (may already be dead)"
        fi
    done
    
    # Wait for processes to die
    sleep 2
    
    # Verify they're dead
    REMAINING=$(ps -aux | grep "test_cm_vec2vec.py" | grep -v grep | wc -l)
    if [ $REMAINING -gt 0 ]; then
        echo "Warning: Some processes may still be running"
    else
        echo "All test_cm_vec2vec.py processes successfully terminated"
    fi
}

# Function to run the new command
run_new_command() {
    echo "Starting new test_cm_vec2vec.py with updated parameters..."
    echo "Command: python test_cm_vec2vec.py --dataset=bpmn --batch_size=8192 --enhanced_losses --save_table --lr_generator=1e-3 --lr_discriminator=0.004 --epochs=100"
    
    python test_cm_vec2vec.py --dataset=bpmn --batch_size=8192 --enhanced_losses --save_table --lr_generator=1e-3 --lr_discriminator=0.004 --epochs=100
}

# Main execution
main() {
    kill_test_processes
    run_new_command
}

# Run the main function
main "$@"