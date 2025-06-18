# Signal Handling and Graceful Shutdown

## Overview

The training system now includes comprehensive SIGINT (Ctrl-C) handling to ensure graceful shutdown and proper cleanup of all child processes and resources.

## Features

### 1. **Global Signal Handler**
- Catches SIGINT (Ctrl-C) signals
- Manages shutdown of all active components
- Prevents orphaned processes
- Saves cache and checkpoint data

### 2. **Thread Pool Management**
- Cancels all pending futures
- Shuts down ThreadPoolExecutor gracefully
- Tracks active tasks for proper cleanup

### 3. **Cache Preservation**
- Saves cache manifest before exit
- Uses atomic writes to prevent corruption
- Validates cache on next run

### 4. **Checkpoint Saving**
- Automatically saves checkpoint on interrupt
- Preserves training progress
- Allows resuming from exact point

## Usage

### Normal Training
```bash
python train_indicator_transformer.py

# Press Ctrl-C to interrupt
# System will:
# - Stop current epoch
# - Save checkpoint
# - Clean up processes
# - Save cache data
```

### Resume from Checkpoint
```bash
# List available checkpoints
python resume_training.py --list

# Resume from latest checkpoint
python resume_training.py

# Resume from specific checkpoint
python resume_training.py --checkpoint models/checkpoint_epoch_15_interrupted.pth

# Resume and train 30 more epochs
python resume_training.py --epochs 30
```

## Signal Flow

1. **User presses Ctrl-C**
   ```
   SIGINT → SignalHandler._handle_signal()
   ```

2. **Shutdown sequence initiated**
   - Set `shutdown_requested` flag
   - Cancel active futures
   - Shutdown thread pool
   - Save cache manifest

3. **Training loop checks**
   ```python
   if signal_handler.should_stop:
       # Save checkpoint
       # Exit gracefully
   ```

4. **Cleanup on exit**
   - Restore original signal handler
   - Ensure all resources freed

## Implementation Details

### SignalHandler Class
```python
class SignalHandler:
    - shutdown_requested: bool
    - executor: ThreadPoolExecutor
    - active_futures: List[Future]
    - register_executor(): Track executor
    - register_future(): Track futures
    - should_stop: Check shutdown status
```

### Integration Points

1. **Data Preparation**
   - Checks before submitting tasks
   - Checks during processing
   - Proper future cleanup

2. **Training Epochs**
   - Checks at start of epoch
   - Checks after each batch
   - Saves checkpoint on interrupt

3. **Cache Operations**
   - Atomic writes prevent corruption
   - Manifest saved on shutdown
   - Validation on startup

## Benefits

1. **No Data Loss**
   - Training progress saved
   - Cache data preserved
   - Clean checkpoint files

2. **Clean Process Management**
   - No orphaned processes
   - Proper thread cleanup
   - Resource deallocation

3. **Easy Recovery**
   - Resume from exact point
   - Restore optimizer state
   - Continue loss history

## Troubleshooting

### Corrupted Cache Files
```bash
# Validate and clean cache
python validate_cache.py

# Clear all cache if needed
python validate_cache.py --clear-all
```

### Can't Find Checkpoint
```bash
# List all checkpoints
python resume_training.py --list

# Check models directory
ls -la models/*.pth
```

### Process Still Running
```bash
# Check for Python processes
ps aux | grep python

# Kill if needed (last resort)
kill -9 <PID>
```

## Best Practices

1. **Let cleanup finish**
   - Wait for "Cleanup complete" message
   - Don't force kill unless necessary

2. **Check logs**
   - Review shutdown messages
   - Verify checkpoint saved

3. **Resume promptly**
   - Checkpoints include timestamp
   - Resume soon to continue momentum

4. **Monitor resources**
   - Check process list after shutdown
   - Verify memory freed

## Technical Notes

- Uses Python's `signal` module
- Thread-safe with locks
- Atomic file operations
- Compatible with multiprocessing
- Works on macOS/Linux/Windows (with limitations)