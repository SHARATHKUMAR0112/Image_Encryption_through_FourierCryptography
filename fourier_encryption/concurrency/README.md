# Concurrency Module

This module provides thread-safe components for concurrent encryption of multiple images, enabling parallel processing and improved performance.

## Components

### ConcurrentEncryptionOrchestrator

Thread-safe orchestrator for parallel image encryption using a thread pool.

**Features:**
- Thread pool for parallel processing (configurable worker count)
- Thread-safe result collection and storage
- Progress tracking across multiple tasks
- Error handling and recovery
- Resource cleanup

**Usage:**

```python
from fourier_encryption.concurrency import (
    ConcurrentEncryptionOrchestrator,
    EncryptionTask,
)

# Create concurrent orchestrator
concurrent_orch = ConcurrentEncryptionOrchestrator(
    orchestrator=base_orchestrator,
    max_workers=4,  # Use 4 worker threads
    progress_callback=lambda task_id, progress: print(f"{task_id}: {progress}%"),
)

# Prepare tasks
tasks = [
    EncryptionTask(
        image_path=Path("image1.png"),
        key="password1",
        preprocess_config=PreprocessConfig(),
        encryption_config=EncryptionConfig(),
        task_id="task_1",
    ),
    # ... more tasks
]

# Execute batch (blocks until all complete)
results = concurrent_orch.encrypt_batch(tasks, wait=True)

# Check results
for result in results:
    if result.success:
        print(f"{result.task_id}: Success")
    else:
        print(f"{result.task_id}: Failed - {result.error}")
```

### ThreadSafeRenderer

Thread-safe wrapper for LiveRenderer, ensuring synchronized access to visualization operations.

**Features:**
- Thread-safe frame rendering
- Synchronized observer notifications
- Proper locking for shared state
- Safe concurrent access to trace points
- Reentrant lock for nested calls

**Usage:**

```python
from fourier_encryption.concurrency import ThreadSafeRenderer

# Create thread-safe renderer
renderer = ThreadSafeRenderer(backend="matplotlib")

# Use from multiple threads safely
def render_in_thread(state, frame_num):
    renderer.render_frame(state, frame_num)

# Multiple threads can call render_frame concurrently
threads = [
    threading.Thread(target=render_in_thread, args=(state, i))
    for i in range(10)
]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()
```

## Thread Safety Guarantees

### ConcurrentEncryptionOrchestrator

- **Result Storage**: All result access is protected by `_results_lock`
- **Task ID Generation**: Task counter is protected by `_task_counter_lock`
- **Thread Pool**: Uses Python's `ThreadPoolExecutor` for safe parallel execution
- **Progress Callbacks**: Callbacks are invoked from worker threads but don't block main thread

### ThreadSafeRenderer

- **Rendering Operations**: All rendering operations are protected by `_render_lock`
- **Observer Notifications**: Observer list access is synchronized
- **Trace Points**: Trace point access returns a copy to prevent concurrent modification
- **Reentrant Lock**: Uses `threading.RLock` to allow nested calls from the same thread

## Performance Considerations

### Thread Pool Sizing

The optimal number of worker threads depends on:
- **CPU-bound tasks**: Set `max_workers` to CPU count (default)
- **I/O-bound tasks**: Can use more workers (e.g., 2x CPU count)
- **Memory constraints**: More workers = more memory usage

```python
import os

# CPU-bound (default)
max_workers = None  # Uses os.cpu_count()

# I/O-bound
max_workers = os.cpu_count() * 2

# Memory-constrained
max_workers = 2
```

### Batch Size

For large batches, consider:
- **Memory usage**: Each task holds image data in memory
- **Progress tracking**: Smaller batches provide more frequent updates
- **Error recovery**: Smaller batches isolate failures

```python
# Process in smaller batches
batch_size = 10
for i in range(0, len(all_tasks), batch_size):
    batch = all_tasks[i:i+batch_size]
    results = concurrent_orch.encrypt_batch(batch, wait=True)
```

## Error Handling

### Task-Level Errors

Individual task failures don't affect other tasks:

```python
results = concurrent_orch.encrypt_batch(tasks, wait=True)

successful = [r for r in results if r.success]
failed = [r for r in results if not r.success]

print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")

# Retry failed tasks
retry_tasks = [
    EncryptionTask(
        image_path=r.image_path,
        key="new_password",
        preprocess_config=PreprocessConfig(),
        encryption_config=EncryptionConfig(),
    )
    for r in failed
]
```

### Rendering Errors

Rendering errors are caught and logged but don't crash the thread:

```python
# Even if rendering fails, the thread continues
renderer.render_frame(invalid_state, frame_number=0)
# Error is logged, but no exception is raised
```

## Examples

See `examples/concurrent_encryption_demo.py` for a complete demonstration.

## Testing

Unit tests are available in:
- `tests/unit/test_concurrent_orchestrator.py`
- `tests/unit/test_thread_safe_renderer.py`

Run tests:
```bash
pytest tests/unit/test_concurrent_orchestrator.py -v
pytest tests/unit/test_thread_safe_renderer.py -v
```

## Requirements

Satisfies acceptance criteria:
- **AC 3.13.3**: System shall support concurrent encryption of multiple images
- **AC 3.13.4**: Rendering shall be thread-safe for parallel operations
