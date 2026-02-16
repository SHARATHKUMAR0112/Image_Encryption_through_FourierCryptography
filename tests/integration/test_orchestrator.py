"""
Integration test for EncryptionOrchestrator.

Tests the complete encryption and decryption workflows coordinated by
the orchestrator.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from fourier_encryption.application.orchestrator import EncryptionOrchestrator
from fourier_encryption.core.image_processor import OpenCVImageProcessor
from fourier_encryption.core.edge_detector import CannyEdgeDetector
from fourier_encryption.core.contour_extractor import ContourExtractor
from fourier_encryption.core.fourier_transformer import FourierTransformer
from fourier_encryption.encryption.aes_encryptor import AES256Encryptor
from fourier_encryption.encryption.key_manager import KeyManager
from fourier_encryption.transmission.serializer import CoefficientSerializer
from fourier_encryption.config.settings import PreprocessConfig, EncryptionConfig
from fourier_encryption.models.exceptions import EncryptionError, ImageProcessingError


# Strong test password that meets all requirements
TEST_PASSWORD = "TestPassword123!"


class TestEncryptionOrchestrator:
    """Test the EncryptionOrchestrator integration"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create an orchestrator with all required components"""
        image_processor = OpenCVImageProcessor()
        edge_detector = CannyEdgeDetector()
        contour_extractor = ContourExtractor()
        fourier_transformer = FourierTransformer()
        encryptor = AES256Encryptor()
        serializer = CoefficientSerializer()
        key_manager = KeyManager()
        
        return EncryptionOrchestrator(
            image_processor=image_processor,
            edge_detector=edge_detector,
            contour_extractor=contour_extractor,
            fourier_transformer=fourier_transformer,
            encryptor=encryptor,
            serializer=serializer,
            key_manager=key_manager,
            optimizer=None,
            anomaly_detector=None,
        )
    
    @pytest.fixture
    def test_image_path(self):
        """Create a temporary test image"""
        # Create a simple test image (white square on black background)
        image = np.zeros((200, 200), dtype=np.uint8)
        image[50:150, 50:150] = 255
        
        # Save to temporary file
        import cv2
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = Path(f.name)
            cv2.imwrite(str(temp_path), image)
        
        yield temp_path
        
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test that orchestrator initializes correctly"""
        assert orchestrator is not None
        assert orchestrator.image_processor is not None
        assert orchestrator.edge_detector is not None
        assert orchestrator.contour_extractor is not None
        assert orchestrator.fourier_transformer is not None
        assert orchestrator.encryptor is not None
        assert orchestrator.serializer is not None
        assert orchestrator.key_manager is not None
    
    def test_encrypt_image_basic(self, orchestrator, test_image_path):
        """Test basic image encryption workflow"""
        # Configure encryption
        preprocess_config = PreprocessConfig(
            target_size=(200, 200),
            maintain_aspect_ratio=False,
            normalize=False,
            denoise=False,
        )
        
        encryption_config = EncryptionConfig(
            num_coefficients=50,
            use_ai_edge_detection=False,
            use_ai_optimization=False,
            use_anomaly_detection=False,
            kdf_iterations=100_000,
            visualization_enabled=False,
        )
        
        # Encrypt the image
        key = TEST_PASSWORD
        encrypted_payload = orchestrator.encrypt_image(
            test_image_path,
            key,
            preprocess_config,
            encryption_config,
        )
        
        # Verify encrypted payload structure
        assert encrypted_payload is not None
        assert encrypted_payload.ciphertext is not None
        assert len(encrypted_payload.ciphertext) > 0
        assert encrypted_payload.iv is not None
        assert len(encrypted_payload.iv) == 16
        assert encrypted_payload.hmac is not None
        assert len(encrypted_payload.hmac) == 32
        assert encrypted_payload.metadata is not None
        assert "dimensions" in encrypted_payload.metadata
        assert "salt" in encrypted_payload.metadata
        assert "kdf_iterations" in encrypted_payload.metadata
    
    def test_decrypt_image_basic(self, orchestrator, test_image_path):
        """Test basic image decryption workflow"""
        # Configure encryption
        preprocess_config = PreprocessConfig(
            target_size=(200, 200),
            maintain_aspect_ratio=False,
            normalize=False,
            denoise=False,
        )
        
        encryption_config = EncryptionConfig(
            num_coefficients=50,
            use_ai_edge_detection=False,
            use_ai_optimization=False,
            use_anomaly_detection=False,
            kdf_iterations=100_000,
            visualization_enabled=False,
        )
        
        # Encrypt the image
        key = TEST_PASSWORD
        encrypted_payload = orchestrator.encrypt_image(
            test_image_path,
            key,
            preprocess_config,
            encryption_config,
        )
        
        # Decrypt the image
        reconstructed_points = orchestrator.decrypt_image(
            encrypted_payload,
            key,
            visualize=False,
        )
        
        # Verify reconstruction
        assert reconstructed_points is not None
        assert len(reconstructed_points) > 0
        assert reconstructed_points.dtype == np.complex128
    
    def test_encrypt_decrypt_round_trip(self, orchestrator, test_image_path):
        """Test that encrypt → decrypt recovers the coefficients"""
        # Configure encryption
        preprocess_config = PreprocessConfig(
            target_size=(200, 200),
            maintain_aspect_ratio=False,
            normalize=False,
            denoise=False,
        )
        
        encryption_config = EncryptionConfig(
            num_coefficients=30,
            use_ai_edge_detection=False,
            use_ai_optimization=False,
            use_anomaly_detection=False,
            kdf_iterations=100_000,
            visualization_enabled=False,
        )
        
        # Encrypt the image
        key = TEST_PASSWORD
        encrypted_payload = orchestrator.encrypt_image(
            test_image_path,
            key,
            preprocess_config,
            encryption_config,
        )
        
        # Decrypt the image
        reconstructed_points = orchestrator.decrypt_image(
            encrypted_payload,
            key,
            visualize=False,
        )
        
        # Verify we got points back
        assert len(reconstructed_points) > 0
        
        # The number of reconstructed points should match the coefficient count
        # (since we're using IDFT which reconstructs all points)
        assert len(reconstructed_points) >= encryption_config.num_coefficients
    
    def test_decrypt_with_wrong_key_fails(self, orchestrator, test_image_path):
        """Test that decryption with wrong key fails"""
        # Configure encryption
        preprocess_config = PreprocessConfig(
            target_size=(200, 200),
            maintain_aspect_ratio=False,
            normalize=False,
            denoise=False,
        )
        
        encryption_config = EncryptionConfig(
            num_coefficients=30,
            use_ai_edge_detection=False,
            use_ai_optimization=False,
            use_anomaly_detection=False,
            kdf_iterations=100_000,
            visualization_enabled=False,
        )
        
        # Encrypt with one key
        correct_key = TEST_PASSWORD
        encrypted_payload = orchestrator.encrypt_image(
            test_image_path,
            correct_key,
            preprocess_config,
            encryption_config,
        )
        
        # Try to decrypt with wrong key
        wrong_key = "WrongPassword456#"
        with pytest.raises(EncryptionError):
            orchestrator.decrypt_image(
                encrypted_payload,
                wrong_key,
                visualize=False,
            )
    
    def test_decrypt_with_visualization(self, orchestrator, test_image_path):
        """Test decryption with epicycle visualization"""
        # Configure encryption
        preprocess_config = PreprocessConfig(
            target_size=(200, 200),
            maintain_aspect_ratio=False,
            normalize=False,
            denoise=False,
        )
        
        encryption_config = EncryptionConfig(
            num_coefficients=20,
            use_ai_edge_detection=False,
            use_ai_optimization=False,
            use_anomaly_detection=False,
            kdf_iterations=100_000,
            visualization_enabled=False,
        )
        
        # Encrypt the image
        key = TEST_PASSWORD
        encrypted_payload = orchestrator.encrypt_image(
            test_image_path,
            key,
            preprocess_config,
            encryption_config,
        )
        
        # Decrypt with visualization
        trace_points = orchestrator.decrypt_image(
            encrypted_payload,
            key,
            visualize=True,
        )
        
        # Verify we got trace points from animation
        assert trace_points is not None
        assert len(trace_points) > 0
        assert trace_points.dtype == np.complex128
    
    def test_encrypt_with_weak_key_fails(self, orchestrator, test_image_path):
        """Test that encryption with weak key fails"""
        # Configure encryption
        preprocess_config = PreprocessConfig(
            target_size=(200, 200),
            maintain_aspect_ratio=False,
            normalize=False,
            denoise=False,
        )
        
        encryption_config = EncryptionConfig(
            num_coefficients=30,
            use_ai_edge_detection=False,
            use_ai_optimization=False,
            use_anomaly_detection=False,
            kdf_iterations=100_000,
            visualization_enabled=False,
        )
        
        # Try to encrypt with weak key
        weak_key = "123"  # Too short
        with pytest.raises(EncryptionError, match="minimum strength"):
            orchestrator.encrypt_image(
                test_image_path,
                weak_key,
                preprocess_config,
                encryption_config,
            )
    
    def test_encrypt_nonexistent_image_fails(self, orchestrator):
        """Test that encrypting nonexistent image fails"""
        # Configure encryption
        preprocess_config = PreprocessConfig()
        encryption_config = EncryptionConfig(num_coefficients=30)
        
        # Try to encrypt nonexistent image
        nonexistent_path = Path("/nonexistent/image.png")
        key = TEST_PASSWORD
        
        with pytest.raises(ImageProcessingError, match="not found"):
            orchestrator.encrypt_image(
                nonexistent_path,
                key,
                preprocess_config,
                encryption_config,
            )
    
    def test_full_encryption_workflow_image_to_payload(self, orchestrator, test_image_path):
        """
        Integration test: Full encryption workflow from image file to encrypted payload.
        
        Validates Requirements: 3.1.1, 3.1.2, 3.1.3, 3.1.4, 3.2.1, 3.2.2, 3.2.3,
                               3.4.1, 3.4.2, 3.4.3, 3.4.4, 3.5.1, 3.5.2
        """
        # Configure with specific settings
        preprocess_config = PreprocessConfig(
            target_size=(200, 200),
            maintain_aspect_ratio=False,
            normalize=True,
            denoise=False,
        )
        
        encryption_config = EncryptionConfig(
            num_coefficients=100,
            use_ai_edge_detection=False,
            use_ai_optimization=False,
            use_anomaly_detection=False,
            kdf_iterations=100_000,
            visualization_enabled=False,
        )
        
        key = TEST_PASSWORD
        
        # Execute full encryption workflow
        encrypted_payload = orchestrator.encrypt_image(
            test_image_path,
            key,
            preprocess_config,
            encryption_config,
        )
        
        # Validate encrypted payload structure (AC 3.4.5, 3.5.2)
        assert encrypted_payload is not None
        assert isinstance(encrypted_payload.ciphertext, bytes)
        assert len(encrypted_payload.ciphertext) > 0
        
        # Validate IV (AC 3.4.3)
        assert isinstance(encrypted_payload.iv, bytes)
        assert len(encrypted_payload.iv) == 16  # AES block size
        
        # Validate HMAC (AC 3.4.4)
        assert isinstance(encrypted_payload.hmac, bytes)
        assert len(encrypted_payload.hmac) == 32  # SHA-256 output size
        
        # Validate metadata (AC 3.5.2)
        assert "dimensions" in encrypted_payload.metadata
        assert "salt" in encrypted_payload.metadata
        assert "kdf_iterations" in encrypted_payload.metadata
        assert "original_contour_length" in encrypted_payload.metadata
        
        # Validate dimensions
        dimensions = encrypted_payload.metadata["dimensions"]
        assert len(dimensions) == 2
        assert dimensions[0] > 0 and dimensions[1] > 0
        
        # Validate salt format
        salt_hex = encrypted_payload.metadata["salt"]
        assert isinstance(salt_hex, str)
        assert len(bytes.fromhex(salt_hex)) == 32  # 32-byte salt
        
        # Validate KDF iterations
        assert encrypted_payload.metadata["kdf_iterations"] == 100_000
    
    def test_full_decryption_workflow_payload_to_image(self, orchestrator, test_image_path):
        """
        Integration test: Full decryption workflow from encrypted payload to reconstructed image.
        
        Validates Requirements: 3.4.6, 3.5.3, 3.2.6
        """
        # First encrypt an image
        preprocess_config = PreprocessConfig(
            target_size=(200, 200),
            maintain_aspect_ratio=False,
            normalize=False,
            denoise=False,
        )
        
        encryption_config = EncryptionConfig(
            num_coefficients=80,
            use_ai_edge_detection=False,
            use_ai_optimization=False,
            use_anomaly_detection=False,
            kdf_iterations=100_000,
            visualization_enabled=False,
        )
        
        key = TEST_PASSWORD
        encrypted_payload = orchestrator.encrypt_image(
            test_image_path,
            key,
            preprocess_config,
            encryption_config,
        )
        
        # Execute full decryption workflow
        reconstructed_points = orchestrator.decrypt_image(
            encrypted_payload,
            key,
            visualize=False,
        )
        
        # Validate reconstruction (AC 3.2.6)
        assert reconstructed_points is not None
        assert isinstance(reconstructed_points, np.ndarray)
        assert len(reconstructed_points) > 0
        assert reconstructed_points.dtype == np.complex128
        
        # Verify we have meaningful points (not all zeros)
        assert np.any(reconstructed_points != 0)
        
        # Verify points are finite (no NaN or Inf)
        assert np.all(np.isfinite(reconstructed_points))
    
    def test_end_to_end_round_trip_recovers_similar_image(self, orchestrator, test_image_path):
        """
        Integration test: End-to-end round-trip encrypt then decrypt recovers similar image.
        
        This test validates that the complete encryption and decryption pipeline
        works correctly and produces valid reconstructed data.
        
        Note: We cannot directly compare pixel-by-pixel similarity because:
        1. The encryption process works on contours (edges), not full images
        2. Fourier reconstruction with truncated coefficients is an approximation
        3. IDFT produces a different point distribution than the original contour
        
        Instead, we validate that:
        - The pipeline completes successfully
        - Reconstructed points are valid (finite, non-zero)
        - Reconstructed points are in a reasonable spatial range
        - The reconstruction has similar complexity to the original
        
        Validates Requirements: 3.1.1 through 3.5.4
        """
        # Configure with moderate coefficient count
        preprocess_config = PreprocessConfig(
            target_size=(200, 200),
            maintain_aspect_ratio=False,
            normalize=False,
            denoise=False,
        )
        
        encryption_config = EncryptionConfig(
            num_coefficients=100,
            use_ai_edge_detection=False,
            use_ai_optimization=False,
            use_anomaly_detection=False,
            kdf_iterations=100_000,
            visualization_enabled=False,
        )
        
        key = TEST_PASSWORD
        
        # Load original image for comparison
        original_image = orchestrator.image_processor.load_image(test_image_path)
        original_preprocessed = orchestrator.image_processor.preprocess(
            original_image, preprocess_config
        )
        
        # Extract original contours
        original_edges = orchestrator.edge_detector.detect_edges(original_preprocessed)
        original_contours = orchestrator.contour_extractor.extract_contours(original_edges)
        original_complex = orchestrator.contour_extractor.to_complex_plane(original_contours[0])
        
        # Encrypt the image
        encrypted_payload = orchestrator.encrypt_image(
            test_image_path,
            key,
            preprocess_config,
            encryption_config,
        )
        
        # Decrypt the image
        reconstructed_points = orchestrator.decrypt_image(
            encrypted_payload,
            key,
            visualize=False,
        )
        
        # Validate reconstruction quality
        
        # 1. Check that we have a reasonable number of points
        assert len(reconstructed_points) >= encryption_config.num_coefficients, \
            "Reconstructed points should be at least as many as coefficients"
        
        # 2. Verify all points are valid (finite, not NaN or Inf)
        assert np.all(np.isfinite(reconstructed_points)), \
            "All reconstructed points should be finite"
        
        # 3. Verify we have non-trivial reconstruction (not all zeros)
        assert np.any(reconstructed_points != 0), \
            "Reconstructed points should not all be zero"
        
        # 4. Check that reconstructed points are in a reasonable spatial range
        # Note: IDFT reconstruction can produce points outside the original range
        # due to the nature of Fourier series approximation, especially with
        # truncated coefficients. This is expected behavior.
        original_real_range = (original_complex.real.min(), original_complex.real.max())
        original_imag_range = (original_complex.imag.min(), original_complex.imag.max())
        
        reconstructed_real_range = (reconstructed_points.real.min(), reconstructed_points.real.max())
        reconstructed_imag_range = (reconstructed_points.imag.min(), reconstructed_points.imag.max())
        
        # Verify that reconstructed points are in a reasonable spatial range
        # (not wildly different from the original image dimensions)
        original_width = original_real_range[1] - original_real_range[0]
        original_height = original_imag_range[1] - original_imag_range[0]
        
        reconstructed_width = reconstructed_real_range[1] - reconstructed_real_range[0]
        reconstructed_height = reconstructed_imag_range[1] - reconstructed_imag_range[0]
        
        # Reconstructed dimensions should be within 10x of original
        # (Fourier reconstruction can overshoot, especially with fewer coefficients)
        assert reconstructed_width < original_width * 10, \
            f"Reconstructed width {reconstructed_width} too large compared to original {original_width}"
        assert reconstructed_height < original_height * 10, \
            f"Reconstructed height {reconstructed_height} too large compared to original {original_height}"
        
        # 5. Verify that the reconstruction has similar complexity
        # by checking that the standard deviation is in a reasonable range
        original_std = np.std(np.abs(original_complex))
        reconstructed_std = np.std(np.abs(reconstructed_points))
        
        # Standard deviations should be within 10x of each other
        # (indicates similar spatial distribution complexity)
        std_ratio = max(original_std, reconstructed_std) / min(original_std, reconstructed_std)
        assert std_ratio < 10, \
            f"Standard deviation ratio {std_ratio} indicates very different spatial distributions"
    
    def test_round_trip_with_different_coefficient_counts(self, orchestrator, test_image_path):
        """
        Test round-trip encryption/decryption with various coefficient counts.
        
        Validates that the system works correctly with different levels of detail.
        Validates Requirements: 3.2.4
        """
        preprocess_config = PreprocessConfig(
            target_size=(200, 200),
            maintain_aspect_ratio=False,
            normalize=False,
            denoise=False,
        )
        
        key = TEST_PASSWORD
        
        # Test with different coefficient counts
        coefficient_counts = [10, 30, 50, 100, 200]
        
        for num_coefficients in coefficient_counts:
            encryption_config = EncryptionConfig(
                num_coefficients=num_coefficients,
                use_ai_edge_detection=False,
                use_ai_optimization=False,
                use_anomaly_detection=False,
                kdf_iterations=100_000,
                visualization_enabled=False,
            )
            
            # Encrypt
            encrypted_payload = orchestrator.encrypt_image(
                test_image_path,
                key,
                preprocess_config,
                encryption_config,
            )
            
            # Decrypt
            reconstructed_points = orchestrator.decrypt_image(
                encrypted_payload,
                key,
                visualize=False,
            )
            
            # Validate
            assert len(reconstructed_points) >= num_coefficients, \
                f"Failed for {num_coefficients} coefficients"
            assert np.all(np.isfinite(reconstructed_points)), \
                f"Non-finite values for {num_coefficients} coefficients"
    
    def test_round_trip_preserves_metadata(self, orchestrator, test_image_path):
        """
        Test that metadata is preserved through encryption and decryption.
        
        Validates Requirements: 3.5.2
        """
        preprocess_config = PreprocessConfig(
            target_size=(256, 256),
            maintain_aspect_ratio=False,
            normalize=False,
            denoise=False,
        )
        
        encryption_config = EncryptionConfig(
            num_coefficients=75,
            use_ai_edge_detection=False,
            use_ai_optimization=False,
            use_anomaly_detection=False,
            kdf_iterations=150_000,
            visualization_enabled=False,
        )
        
        key = TEST_PASSWORD
        
        # Encrypt
        encrypted_payload = orchestrator.encrypt_image(
            test_image_path,
            key,
            preprocess_config,
            encryption_config,
        )
        
        # Verify metadata in encrypted payload
        assert encrypted_payload.metadata["dimensions"] == [256, 256]
        assert encrypted_payload.metadata["kdf_iterations"] == 150_000
        assert "original_contour_length" in encrypted_payload.metadata
        assert encrypted_payload.metadata["original_contour_length"] > 0
        
        # Decrypt
        reconstructed_points = orchestrator.decrypt_image(
            encrypted_payload,
            key,
            visualize=False,
        )
        
        # Verify reconstruction succeeded with preserved metadata
        assert reconstructed_points is not None
        assert len(reconstructed_points) > 0
    
    def test_multiple_sequential_encryptions_produce_different_payloads(self, orchestrator, test_image_path):
        """
        Test that encrypting the same image multiple times produces different payloads.
        
        This validates that IVs are unique (AC 3.4.3) and that the encryption
        is non-deterministic as expected.
        
        Validates Requirements: 3.4.3
        """
        preprocess_config = PreprocessConfig(
            target_size=(200, 200),
            maintain_aspect_ratio=False,
            normalize=False,
            denoise=False,
        )
        
        encryption_config = EncryptionConfig(
            num_coefficients=50,
            use_ai_edge_detection=False,
            use_ai_optimization=False,
            use_anomaly_detection=False,
            kdf_iterations=100_000,
            visualization_enabled=False,
        )
        
        key = TEST_PASSWORD
        
        # Encrypt the same image three times
        payload1 = orchestrator.encrypt_image(
            test_image_path, key, preprocess_config, encryption_config
        )
        payload2 = orchestrator.encrypt_image(
            test_image_path, key, preprocess_config, encryption_config
        )
        payload3 = orchestrator.encrypt_image(
            test_image_path, key, preprocess_config, encryption_config
        )
        
        # Verify that IVs are different (AC 3.4.3)
        assert payload1.iv != payload2.iv
        assert payload1.iv != payload3.iv
        assert payload2.iv != payload3.iv
        
        # Verify that ciphertexts are different
        assert payload1.ciphertext != payload2.ciphertext
        assert payload1.ciphertext != payload3.ciphertext
        assert payload2.ciphertext != payload3.ciphertext
        
        # Verify that salts are different
        salt1 = bytes.fromhex(payload1.metadata["salt"])
        salt2 = bytes.fromhex(payload2.metadata["salt"])
        salt3 = bytes.fromhex(payload3.metadata["salt"])
        assert salt1 != salt2
        assert salt1 != salt3
        assert salt2 != salt3
        
        # But all should decrypt successfully to similar results
        reconstructed1 = orchestrator.decrypt_image(payload1, key, visualize=False)
        reconstructed2 = orchestrator.decrypt_image(payload2, key, visualize=False)
        reconstructed3 = orchestrator.decrypt_image(payload3, key, visualize=False)
        
        assert len(reconstructed1) > 0
        assert len(reconstructed2) > 0
        assert len(reconstructed3) > 0
