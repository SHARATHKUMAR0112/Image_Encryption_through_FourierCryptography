"""
Thread pool for concurrent image encryption.

This module provides a thread pool implementation for processing multiple
images in parallel, with proper resource management and error handling.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
from typing import List, Dict, Callable, Optional, Any, Tuple
from dataclasses import dataclass

from fourier_encryption.config.settings import EncryptionConfig, PreprocessConfig
from fourier_encryption.models.data_models import EncryptedPayload
from fourier_encryption.models.exceptions import EncryptionError


logger = logging.getLogger(__name__)


@dataclass
class EncryptionTask:
    """
    Represents a single encryption task.
    
    Attributes:
        image_path: Path to the image file to encrypt
        key: Encryption key/password
        preprocess_config: Image preprocessing configuration
        encryption_config: Encryption configuration
        task_id: Unique identifier for this task
    """
    image_path: Path
    key: str
    preprocess_config: PreprocessConfig
    encryption_config: EncryptionConfig
    task_id: str


@dataclass
class EncryptionResult:
    """
    Result of an encryption task.
    
    Attributes:
        task_id: Unique identifier for the task
        image_path: Path to the original image
        payload: Encrypted payload (if successful)
        error: Error message (if failed)
        success: Whether the encryption succeeded
    """
    task_id: str
    image_path: Path
    payload: Optional[EncryptedPayload] = None
    error: Optional[str] = None
    success: bool = True


class EncryptionThreadPool:
    """
    Thread pool for concurrent image encryption.
    
    Provides a high-level interface for encrypting multiple images in parallel
    using a thread pool. Handles resource management, error handling, and
    progress tracking.
    
    Design Pattern: Thread Pool Pattern
    - Manages a pool of worker threads for parallel processing
    - Queues tasks and distributes them to available workers
    - Provides result collection and error handling
    
    Thread Safety:
    - Uses ThreadPoolExecutor for thread management
    - Each task operates on independent data
    - Shared resources (orchestrator) are accessed through thread-safe methods
    
    Attributes:
        max_workers: Maximum number of concurrent worker threads
        executor: ThreadPoolExecutor instance
        active_tasks: Dictionary of active futures
        results_lock: Lock for thread-safe result collection
    """
    
    def __init__(
        self,
        orchestrator_factory: Callable[[], Any],
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the encryption thread pool.
        
        Args:
            orchestrator_factory: Factory function that creates a new orchestrator
                                 instance for each worker thread
            max_workers: Maximum number of concurrent workers (default: CPU count)
        """
        self.orchestrator_factory = orchestrator_factory
        self.max_workers = max_workers
        self.executor: Optional[ThreadPoolExecutor] = None
        self.active_tasks: Dict[str, Future] = {}
        self.results_lock = threading.Lock()
        
        logger.info(
            "EncryptionThreadPool initialized",
            extra={"max_workers": max_workers or "auto"}
        )
    
    def __enter__(self):
        """Context manager entry - start the thread pool."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - shutdown the thread pool."""
        self.shutdown(wait=True)
        return False
    
    def start(self) -> None:
        """
        Start the thread pool.
        
        Creates and initializes the ThreadPoolExecutor with the configured
        number of workers.
        """
        if self.executor is not None:
            logger.warning("Thread pool already started")
            return
        
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="EncryptionWorker"
        )
        
        logger.info(
            "Thread pool started",
            extra={"max_workers": self.executor._max_workers}
        )
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the thread pool.
        
        Args:
            wait: If True, wait for all pending tasks to complete
        """
        if self.executor is None:
            return
        
        logger.info(
            "Shutting down thread pool",
            extra={"wait": wait, "active_tasks": len(self.active_tasks)}
        )
        
        self.executor.shutdown(wait=wait)
        self.executor = None
        self.active_tasks.clear()
        
        logger.info("Thread pool shutdown complete")
    
    def submit_task(self, task: EncryptionTask) -> Future:
        """
        Submit an encryption task to the thread pool.
        
        Args:
            task: EncryptionTask to execute
        
        Returns:
            Future representing the pending task
        
        Raises:
            RuntimeError: If thread pool is not started
        """
        if self.executor is None:
            raise RuntimeError("Thread pool not started. Call start() first.")
        
        logger.debug(
            "Submitting encryption task",
            extra={"task_id": task.task_id, "image_path": str(task.image_path)}
        )
        
        future = self.executor.submit(self._execute_task, task)
        
        with self.results_lock:
            self.active_tasks[task.task_id] = future
        
        return future
    
    def _execute_task(self, task: EncryptionTask) -> EncryptionResult:
        """
        Execute a single encryption task.
        
        This method runs in a worker thread. Each worker gets its own
        orchestrator instance to avoid shared state issues.
        
        Args:
            task: EncryptionTask to execute
        
        Returns:
            EncryptionResult with the outcome
        """
        logger.debug(
            "Executing encryption task",
            extra={
                "task_id": task.task_id,
                "thread": threading.current_thread().name
            }
        )
        
        try:
            # Create a new orchestrator instance for this worker thread
            orchestrator = self.orchestrator_factory()
            
            # Execute the encryption
            payload = orchestrator.encrypt_image(
                image_path=task.image_path,
                key=task.key,
                preprocess_config=task.preprocess_config,
                encryption_config=task.encryption_config,
            )
            
            logger.debug(
                "Encryption task completed successfully",
                extra={"task_id": task.task_id}
            )
            
            return EncryptionResult(
                task_id=task.task_id,
                image_path=task.image_path,
                payload=payload,
                success=True
            )
            
        except Exception as e:
            logger.error(
                "Encryption task failed",
                extra={
                    "task_id": task.task_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            
            return EncryptionResult(
                task_id=task.task_id,
                image_path=task.image_path,
                error=str(e),
                success=False
            )
        finally:
            # Clean up task from active tasks
            with self.results_lock:
                self.active_tasks.pop(task.task_id, None)
    
    def encrypt_batch(
        self,
        tasks: List[EncryptionTask],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[EncryptionResult]:
        """
        Encrypt multiple images in parallel.
        
        Submits all tasks to the thread pool and waits for completion,
        collecting results as they finish.
        
        Args:
            tasks: List of EncryptionTask objects to process
            progress_callback: Optional callback(completed, total) for progress updates
        
        Returns:
            List of EncryptionResult objects in completion order
        
        Raises:
            RuntimeError: If thread pool is not started
        """
        if self.executor is None:
            raise RuntimeError("Thread pool not started. Call start() first.")
        
        if not tasks:
            logger.warning("No tasks to process")
            return []
        
        logger.info(
            "Starting batch encryption",
            extra={"task_count": len(tasks)}
        )
        
        # Submit all tasks
        futures = {self.submit_task(task): task for task in tasks}
        
        # Collect results as they complete
        results = []
        completed = 0
        total = len(tasks)
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total)
                
                logger.debug(
                    "Task completed",
                    extra={
                        "completed": completed,
                        "total": total,
                        "success": result.success
                    }
                )
                
            except Exception as e:
                # This should not happen as exceptions are caught in _execute_task
                # But we handle it defensively
                task = futures[future]
                logger.error(
                    "Unexpected error collecting task result",
                    extra={
                        "task_id": task.task_id,
                        "error": str(e)
                    }
                )
                
                results.append(EncryptionResult(
                    task_id=task.task_id,
                    image_path=task.image_path,
                    error=f"Unexpected error: {e}",
                    success=False
                ))
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total)
        
        # Log summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        logger.info(
            "Batch encryption complete",
            extra={
                "total": total,
                "successful": successful,
                "failed": failed
            }
        )
        
        return results
    
    def get_active_task_count(self) -> int:
        """
        Get the number of currently active tasks.
        
        Returns:
            Number of tasks currently being processed
        """
        with self.results_lock:
            return len(self.active_tasks)
    
    def cancel_all_tasks(self) -> int:
        """
        Cancel all pending tasks.
        
        Attempts to cancel all tasks that haven't started execution yet.
        Tasks that are already running will complete normally.
        
        Returns:
            Number of tasks successfully cancelled
        """
        cancelled_count = 0
        
        with self.results_lock:
            for task_id, future in list(self.active_tasks.items()):
                if future.cancel():
                    cancelled_count += 1
                    logger.debug(
                        "Task cancelled",
                        extra={"task_id": task_id}
                    )
        
        logger.info(
            "Task cancellation complete",
            extra={"cancelled": cancelled_count}
        )
        
        return cancelled_count
