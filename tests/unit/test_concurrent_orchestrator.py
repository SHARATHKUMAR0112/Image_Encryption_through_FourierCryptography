"""
Unit tests for ConcurrentEncryptionOrchestrator.

Tests concurrent encryption capabilities, thread safety, and result handling.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock

from fourier_encryption.concurrency import (
    ConcurrentEncryptionOrchestrator,
    EncryptionTask,
    EncryptionResult,
)
from fourier_encryption.config.settings import EncryptionConfig, PreprocessConfig
from fourier_encryption.models.data_models import EncryptedPayload


class TestConcurrentEncryptionOrchestrator:
    """Test suite for ConcurrentEncryptionOrchestrator."""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock base orchestrator."""
        orchestrator = Mock()
        
        # Mock successful encryption (IV must be exactly 16 bytes, HMAC must be 32 bytes)
        mock_payload = EncryptedPayload(
            ciphertext=b"encrypted_data",
            iv=b"1234567890123456",  # Exactly 16 bytes for AES
            hmac=b"12345678901234567890123456789012",  # Exactly 32 bytes for SHA-256
            metadata={"test": "data"},
        )
        orchestrator.encrypt_image.return_value = mock_payload
        
        return orchestrator
    
    @pytest.fixture
    def concurrent_orchestrator(self, mock_orchestrator):
        """Create a ConcurrentEncryptionOrchestrator instance."""
        return ConcurrentEncryptionOrchestrator(
            orchestrator=mock_orchestrator,
            max_workers=2,
        )
    
    def test_initialization(self, mock_orchestrator):
        """Test orchestrator initialization."""
        orchestrator = ConcurrentEncryptionOrchestrator(
            orchestrator=mock_orchestrator,
            max_workers=4,
        )
        
        assert orchestrator.orchestrator == mock_orchestrator
        assert orchestrator.max_workers == 4
        assert orchestrator._task_counter == 0
    
    def test_get_next_task_id(self, concurrent_orchestrator):
        """Test task ID generation."""
        task_id_1 = concurrent_orchestrator._get_next_task_id()
        task_id_2 = concurrent_orchestrator._get_next_task_id()
        
        assert task_id_1 == "task_1"
        assert task_id_2 == "task_2"
        assert task_id_1 != task_id_2
    
    def test_encrypt_single_task_success(self, concurrent_orchestrator, mock_orchestrator):
        """Test successful encryption of a single task."""
        task = EncryptionTask(
            image_path=Path("test.png"),
            key="test_password",
            preprocess_config=PreprocessConfig(),
            encryption_config=EncryptionConfig(),
            task_id="test_task_1",
        )
        
        result = concurrent_orchestrator._encrypt_single_task(task)
        
        assert result.success is True
        assert result.task_id == "test_task_1"
        assert result.image_path == Path("test.png")
        assert result.payload is not None
        assert result.error is None
        
        # Verify orchestrator was called
        mock_orchestrator.encrypt_image.assert_called_once()
    
    def test_encrypt_single_task_failure(self, concurrent_orchestrator, mock_orchestrator):
        """Test failed encryption of a single task."""
        # Make orchestrator raise an exception
        mock_orchestrator.encrypt_image.side_effect = Exception("Encryption failed")
        
        task = EncryptionTask(
            image_path=Path("test.png"),
            key="test_password",
            preprocess_config=PreprocessConfig(),
            encryption_config=EncryptionConfig(),
            task_id="test_task_2",
        )
        
        result = concurrent_orchestrator._encrypt_single_task(task)
        
        assert result.success is False
        assert result.task_id == "test_task_2"
        assert result.payload is None
        assert result.error is not None
        assert "Encryption failed" in result.error
    
    def test_encrypt_batch_empty(self, concurrent_orchestrator):
        """Test batch encryption with empty task list."""
        results = concurrent_orchestrator.encrypt_batch([])
        
        assert results == []
    
    def test_encrypt_batch_single_task(self, concurrent_orchestrator, mock_orchestrator):
        """Test batch encryption with a single task."""
        task = EncryptionTask(
            image_path=Path("test.png"),
            key="test_password",
            preprocess_config=PreprocessConfig(),
            encryption_config=EncryptionConfig(),
        )
        
        results = concurrent_orchestrator.encrypt_batch([task], wait=True)
        
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].payload is not None
    
    def test_encrypt_batch_multiple_tasks(self, concurrent_orchestrator, mock_orchestrator):
        """Test batch encryption with multiple tasks."""
        tasks = [
            EncryptionTask(
                image_path=Path(f"test_{i}.png"),
                key=f"password_{i}",
                preprocess_config=PreprocessConfig(),
                encryption_config=EncryptionConfig(),
                task_id=f"task_{i}",
            )
            for i in range(5)
        ]
        
        results = concurrent_orchestrator.encrypt_batch(tasks, wait=True)
        
        assert len(results) == 5
        assert all(r.success for r in results)
        assert all(r.payload is not None for r in results)
        
        # Verify all tasks were processed
        task_ids = {r.task_id for r in results}
        expected_ids = {f"task_{i}" for i in range(5)}
        assert task_ids == expected_ids
    
    def test_get_result(self, concurrent_orchestrator):
        """Test retrieving a specific result."""
        # Manually add a result
        result = EncryptionResult(
            task_id="test_task",
            image_path=Path("test.png"),
            success=True,
        )
        
        with concurrent_orchestrator._results_lock:
            concurrent_orchestrator._results["test_task"] = result
        
        retrieved = concurrent_orchestrator.get_result("test_task")
        
        assert retrieved is not None
        assert retrieved.task_id == "test_task"
        assert retrieved.success is True
    
    def test_get_result_not_found(self, concurrent_orchestrator):
        """Test retrieving a non-existent result."""
        result = concurrent_orchestrator.get_result("nonexistent")
        
        assert result is None
    
    def test_get_all_results(self, concurrent_orchestrator):
        """Test retrieving all results."""
        # Manually add multiple results
        results = {
            "task_1": EncryptionResult(
                task_id="task_1",
                image_path=Path("test1.png"),
                success=True,
            ),
            "task_2": EncryptionResult(
                task_id="task_2",
                image_path=Path("test2.png"),
                success=True,
            ),
        }
        
        with concurrent_orchestrator._results_lock:
            concurrent_orchestrator._results.update(results)
        
        all_results = concurrent_orchestrator.get_all_results()
        
        assert len(all_results) == 2
        assert "task_1" in all_results
        assert "task_2" in all_results
    
    def test_clear_results(self, concurrent_orchestrator):
        """Test clearing all results."""
        # Add some results
        with concurrent_orchestrator._results_lock:
            concurrent_orchestrator._results["task_1"] = EncryptionResult(
                task_id="task_1",
                image_path=Path("test.png"),
                success=True,
            )
        
        concurrent_orchestrator.clear_results()
        
        all_results = concurrent_orchestrator.get_all_results()
        assert len(all_results) == 0
    
    def test_progress_callback(self, mock_orchestrator):
        """Test progress callback is invoked."""
        callback_calls = []
        
        def progress_callback(task_id: str, progress: float):
            callback_calls.append((task_id, progress))
        
        orchestrator = ConcurrentEncryptionOrchestrator(
            orchestrator=mock_orchestrator,
            max_workers=2,
            progress_callback=progress_callback,
        )
        
        task = EncryptionTask(
            image_path=Path("test.png"),
            key="test_password",
            preprocess_config=PreprocessConfig(),
            encryption_config=EncryptionConfig(),
            task_id="test_task",
        )
        
        orchestrator._encrypt_single_task(task)
        
        # Verify callback was called (at least for start and completion)
        assert len(callback_calls) >= 2
        assert any(progress == 0.0 for _, progress in callback_calls)
        assert any(progress == 100.0 for _, progress in callback_calls)
    
    def test_context_manager(self, concurrent_orchestrator):
        """Test context manager protocol."""
        with concurrent_orchestrator as orch:
            assert orch is concurrent_orchestrator
        
        # Results should be cleared after exit
        all_results = concurrent_orchestrator.get_all_results()
        assert len(all_results) == 0
    
    def test_thread_safety(self, concurrent_orchestrator, mock_orchestrator):
        """Test thread-safe result storage."""
        # Create multiple tasks to run concurrently
        tasks = [
            EncryptionTask(
                image_path=Path(f"test_{i}.png"),
                key=f"password_{i}",
                preprocess_config=PreprocessConfig(),
                encryption_config=EncryptionConfig(),
                task_id=f"task_{i}",
            )
            for i in range(10)
        ]
        
        results = concurrent_orchestrator.encrypt_batch(tasks, wait=True)
        
        # All results should be stored without corruption
        assert len(results) == 10
        
        all_stored_results = concurrent_orchestrator.get_all_results()
        assert len(all_stored_results) == 10
        
        # Verify no duplicate task IDs
        task_ids = [r.task_id for r in results]
        assert len(task_ids) == len(set(task_ids))
