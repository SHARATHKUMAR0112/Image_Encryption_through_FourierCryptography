"""
Concurrent encryption orchestrator for parallel image processing.

This module provides thread-safe concurrent encryption capabilities,
allowing multiple images to be encrypted in parallel using a thread pool.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any

from fourier_encryption.application.orchestrator import EncryptionOrchestrator
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
        task_id: Optional unique identifier for the task
    """
    image_path: Path
    key: str
    preprocess_config: PreprocessConfig
    encryption_config: EncryptionConfig
    task_id: Optional[str] = None


@dataclass
class EncryptionResult:
    """
    Result of an encryption task.
    
    Attributes:
        task_id: Unique identifier for the task
        image_path: Path to the original image
        payload: Encrypted payload (if successful)
        error: Error message (if failed)
        success: Whether encryption succeeded
    """
    task_id: str
    image_path: Path
    payload: Optional[EncryptedPayload] = None
    error: Optional[str] = None
    success: bool = True


class ConcurrentEncryptionOrchestrator:
    """
    Thread-safe orchestrator for concurrent image encryption.
    
    This class manages a thread pool for parallel encryption of multiple images,
    ensuring thread-safe access to shared resources and proper synchronization.
    
    Features:
    - Thread pool for parallel processing
    - Thread-safe result collection
    - Progress tracking across multiple tasks
    - Error handling and recovery
    - Resource cleanup
    
    Attributes:
        orchestrator: Base encryption orchestrator
        max_workers: Maximum number of worker threads
        executor: Thread pool executor
        results_lock: Lock for thread-safe result access
        progress_callback: Optional callback for progress updates
    """
    
    def __init__(
        self,
        orchestrator: EncryptionOrchestrator,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        """
        Initialize the concurrent encryption orchestrator.
        
        Args:
            orchestrator: Base encryption orchestrator instance
            max_workers: Maximum number of worker threads (default: CPU count)
            progress_callback: Optional callback for progress updates
                             Signature: callback(task_id: str, progress: float)
        """
        self.orchestrator = orchestrator
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        
        # Thread pool executor (created on demand)
        self._executor: Optional[ThreadPoolExecutor] = None
        
        # Thread-safe result storage
        self._results: Dict[str, EncryptionResult] = {}
        self._results_lock = threading.Lock()
        
        # Task counter for generating task IDs
        self._task_counter = 0
        self._task_counter_lock = threading.Lock()
        
        logger.info(
            "ConcurrentEncryptionOrchestrator initialized",
            extra={"max_workers": max_workers}
        )
    
    def _get_next_task_id(self) -> str:
        """
        Generate a unique task ID.
        
        Returns:
            Unique task ID string
        """
        with self._task_counter_lock:
            self._task_counter += 1
            return f"task_{self._task_counter}"
    
    def _encrypt_single_task(self, task: EncryptionTask) -> EncryptionResult:
        """
        Encrypt a single image (executed in worker thread).
        
        Args:
            task: Encryption task to execute
        
        Returns:
            EncryptionResult with payload or error
        """
        task_id = task.task_id or self._get_next_task_id()
        
        logger.debug(
            "Starting encryption task",
            extra={"task_id": task_id, "image_path": str(task.image_path)}
        )
        
        try:
            # Report progress: started
            if self.progress_callback:
                self.progress_callback(task_id, 0.0)
            
            # Execute encryption
            payload = self.orchestrator.encrypt_image(
                image_path=task.image_path,
                key=task.key,
                preprocess_config=task.preprocess_config,
                encryption_config=task.encryption_config,
            )
            
            # Report progress: completed
            if self.progress_callback:
                self.progress_callback(task_id, 100.0)
            
            logger.info(
                "Encryption task completed successfully",
                extra={"task_id": task_id}
            )
            
            return EncryptionResult(
                task_id=task_id,
                image_path=task.image_path,
                payload=payload,
                success=True,
            )
            
        except Exception as e:
            logger.error(
                "Encryption task failed",
                extra={
                    "task_id": task_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )
            
            # Report progress: failed
            if self.progress_callback:
                self.progress_callback(task_id, -1.0)
            
            return EncryptionResult(
                task_id=task_id,
                image_path=task.image_path,
                error=str(e),
                success=False,
            )
    
    def encrypt_batch(
        self,
        tasks: List[EncryptionTask],
        wait: bool = True,
    ) -> List[EncryptionResult]:
        """
        Encrypt multiple images concurrently.
        
        Args:
            tasks: List of encryption tasks to execute
            wait: If True, wait for all tasks to complete before returning
        
        Returns:
            List of EncryptionResult objects (in order of completion if wait=True)
        
        Raises:
            RuntimeError: If executor is not properly initialized
        """
        if not tasks:
            logger.warning("encrypt_batch called with empty task list")
            return []
        
        logger.info(
            "Starting batch encryption",
            extra={"num_tasks": len(tasks)}
        )
        
        # Assign task IDs if not provided
        for task in tasks:
            if task.task_id is None:
                task.task_id = self._get_next_task_id()
        
        # Create thread pool executor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures: Dict[Future, EncryptionTask] = {
                executor.submit(self._encrypt_single_task, task): task
                for task in tasks
            }
            
            results: List[EncryptionResult] = []
            
            if wait:
                # Wait for all tasks to complete
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Store result in thread-safe manner
                        with self._results_lock:
                            self._results[result.task_id] = result
                        
                    except Exception as e:
                        logger.error(
                            "Unexpected error retrieving task result",
                            extra={
                                "task_id": task.task_id,
                                "error": str(e),
                            }
                        )
                        
                        # Create error result
                        error_result = EncryptionResult(
                            task_id=task.task_id or "unknown",
                            image_path=task.image_path,
                            error=f"Unexpected error: {e}",
                            success=False,
                        )
                        results.append(error_result)
                        
                        with self._results_lock:
                            self._results[error_result.task_id] = error_result
            else:
                # Return immediately (non-blocking)
                logger.info("Batch encryption started (non-blocking mode)")
        
        logger.info(
            "Batch encryption completed",
            extra={
                "num_tasks": len(tasks),
                "num_successful": sum(1 for r in results if r.success),
                "num_failed": sum(1 for r in results if not r.success),
            }
        )
        
        return results
    
    def get_result(self, task_id: str) -> Optional[EncryptionResult]:
        """
        Get result for a specific task (thread-safe).
        
        Args:
            task_id: Task identifier
        
        Returns:
            EncryptionResult if available, None otherwise
        """
        with self._results_lock:
            return self._results.get(task_id)
    
    def get_all_results(self) -> Dict[str, EncryptionResult]:
        """
        Get all results (thread-safe).
        
        Returns:
            Dictionary mapping task IDs to results
        """
        with self._results_lock:
            return self._results.copy()
    
    def clear_results(self) -> None:
        """
        Clear all stored results (thread-safe).
        """
        with self._results_lock:
            self._results.clear()
            logger.debug("All results cleared")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.clear_results()
        return False
