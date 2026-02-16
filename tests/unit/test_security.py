"""
Unit tests for security module.

Tests input validation, sanitization, and path validation to ensure
security measures prevent injection attacks and information leakage.
"""

import pytest
from pathlib import Path
import tempfile
import os

from fourier_encryption.security.input_validator import InputValidator
from fourier_encryption.security.path_validator import PathValidator
from fourier_encryption.security.sanitizer import Sanitizer


class TestInputValidator:
    """Test input validation for security."""
    
    def test_validate_key_valid(self):
        """Test that valid keys are accepted."""
        assert InputValidator.validate_key("mypassword123")
        assert InputValidator.validate_key("VeryLongPassword123!")
        assert InputValidator.validate_key("a" * 100)
    
    def test_validate_key_too_short(self):
        """Test that short keys are rejected."""
        with pytest.raises(ValueError, match="at least 8 characters"):
            InputValidator.validate_key("short")
    
    def test_validate_key_empty(self):
        """Test that empty keys are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            InputValidator.validate_key("")
    
    def test_validate_key_null_bytes(self):
        """Test that keys with null bytes are rejected."""
        with pytest.raises(ValueError, match="null bytes"):
            InputValidator.validate_key("password\x00malicious")
    
    def test_validate_key_control_characters(self):
        """Test that keys with control characters are rejected."""
        with pytest.raises(ValueError, match="control characters"):
            InputValidator.validate_key("password\x01\x02")
    
    def test_validate_key_injection_patterns(self):
        """Test that keys with injection patterns are rejected."""
        with pytest.raises(ValueError, match="suspicious patterns"):
            InputValidator.validate_key("password'; DROP TABLE users--")
        
        with pytest.raises(ValueError, match="suspicious patterns"):
            InputValidator.validate_key("password && rm -rf /")
    
    def test_validate_coefficient_count_valid(self):
        """Test that valid coefficient counts are accepted."""
        assert InputValidator.validate_coefficient_count(10)
        assert InputValidator.validate_coefficient_count(500)
        assert InputValidator.validate_coefficient_count(1000)
    
    def test_validate_coefficient_count_out_of_range(self):
        """Test that out-of-range coefficient counts are rejected."""
        with pytest.raises(ValueError, match="between 10 and 1000"):
            InputValidator.validate_coefficient_count(5)
        
        with pytest.raises(ValueError, match="between 10 and 1000"):
            InputValidator.validate_coefficient_count(1001)
    
    def test_validate_coefficient_count_wrong_type(self):
        """Test that non-integer coefficient counts are rejected."""
        with pytest.raises(TypeError):
            InputValidator.validate_coefficient_count(10.5)
        
        with pytest.raises(TypeError):
            InputValidator.validate_coefficient_count("10")
    
    def test_validate_animation_speed_valid(self):
        """Test that valid animation speeds are accepted."""
        assert InputValidator.validate_animation_speed(0.1)
        assert InputValidator.validate_animation_speed(1.0)
        assert InputValidator.validate_animation_speed(10.0)
    
    def test_validate_animation_speed_out_of_range(self):
        """Test that out-of-range animation speeds are rejected."""
        with pytest.raises(ValueError, match="between 0.1 and 10.0"):
            InputValidator.validate_animation_speed(0.05)
        
        with pytest.raises(ValueError, match="between 0.1 and 10.0"):
            InputValidator.validate_animation_speed(11.0)
    
    def test_validate_kdf_iterations_valid(self):
        """Test that valid KDF iterations are accepted."""
        assert InputValidator.validate_kdf_iterations(100_000)
        assert InputValidator.validate_kdf_iterations(500_000)
        assert InputValidator.validate_kdf_iterations(10_000_000)
    
    def test_validate_kdf_iterations_out_of_range(self):
        """Test that out-of-range KDF iterations are rejected."""
        with pytest.raises(ValueError, match="between 100000 and 10000000"):
            InputValidator.validate_kdf_iterations(50_000)
        
        with pytest.raises(ValueError, match="between 100000 and 10000000"):
            InputValidator.validate_kdf_iterations(20_000_000)
    
    def test_validate_string_input_valid(self):
        """Test that valid strings are accepted."""
        assert InputValidator.validate_string_input("normal string")
        assert InputValidator.validate_string_input("with-dashes_and_underscores")
    
    def test_validate_string_input_empty(self):
        """Test that empty strings are rejected by default."""
        with pytest.raises(ValueError, match="cannot be empty"):
            InputValidator.validate_string_input("")
        
        # But allowed when allow_empty=True
        assert InputValidator.validate_string_input("", allow_empty=True)
    
    def test_validate_string_input_max_length(self):
        """Test that strings exceeding max length are rejected."""
        with pytest.raises(ValueError, match="exceeds maximum length"):
            InputValidator.validate_string_input("a" * 101, max_length=100)
    
    def test_validate_string_input_injection_patterns(self):
        """Test that strings with injection patterns are rejected."""
        # SQL injection
        with pytest.raises(ValueError, match="suspicious injection patterns"):
            InputValidator.validate_string_input("'; DROP TABLE users--")
        
        # Command injection
        with pytest.raises(ValueError, match="suspicious injection patterns"):
            InputValidator.validate_string_input("test && rm -rf /")
        
        # Path traversal
        with pytest.raises(ValueError, match="suspicious injection patterns"):
            InputValidator.validate_string_input("../../../etc/passwd")
    
    def test_validate_config_dict_valid(self):
        """Test that valid configuration dictionaries are accepted."""
        config = {
            "num_coefficients": 100,
            "animation_speed": 1.0,
            "nested": {
                "key": "value"
            }
        }
        assert InputValidator.validate_config_dict(config)
    
    def test_validate_config_dict_injection_in_values(self):
        """Test that config dicts with injection patterns are rejected."""
        config = {
            "key": "'; DROP TABLE users--"
        }
        with pytest.raises(ValueError, match="suspicious injection patterns"):
            InputValidator.validate_config_dict(config)


class TestPathValidator:
    """Test path validation and traversal prevention."""
    
    def test_validate_path_safe_relative(self):
        """Test that safe relative paths are accepted."""
        path = Path("image.png")
        validated = PathValidator.validate_path(path, must_exist=False)
        assert validated.is_absolute()
    
    def test_validate_path_traversal_rejected(self):
        """Test that path traversal attempts are rejected."""
        with pytest.raises(ValueError, match="dangerous pattern"):
            PathValidator.validate_path(Path("../../../etc/passwd"))
    
    def test_validate_path_absolute_rejected_by_default(self):
        """Test that absolute paths are rejected by default."""
        # On Windows, Path.resolve() always returns absolute paths
        # So we need to test with a path that's clearly outside allowed scope
        with pytest.raises(ValueError, match="dangerous pattern|Absolute paths not allowed"):
            PathValidator.validate_path(Path("C:/Windows/System32/config"))
    
    def test_validate_path_absolute_allowed_when_specified(self):
        """Test that absolute paths are allowed when specified."""
        path = Path("/tmp/test.txt")
        validated = PathValidator.validate_path(path, allow_absolute=True, must_exist=False)
        assert validated.is_absolute()
    
    def test_validate_path_within_base_dir(self):
        """Test that paths are validated to be within base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir).resolve()
            
            # Create a file within the base directory
            safe_path = base_dir / "subdir" / "file.txt"
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            safe_path.touch()
            
            # Should succeed when validating the actual path
            validated = PathValidator.validate_path(
                safe_path,
                base_dir=base_dir,
                must_exist=True,
                allow_absolute=True
            )
            # Verify it's within the base directory
            assert str(validated).startswith(str(base_dir))
    
    def test_validate_image_path_valid_extensions(self):
        """Test that image paths with valid extensions are accepted."""
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            path = Path(f"image{ext}")
            validated = PathValidator.validate_image_path(path, must_exist=False)
            assert validated.suffix.lower() == ext
    
    def test_validate_image_path_invalid_extension(self):
        """Test that image paths with invalid extensions are rejected."""
        with pytest.raises(ValueError, match="Invalid image file extension"):
            PathValidator.validate_image_path(Path("file.txt"), must_exist=False)
    
    def test_validate_payload_path_valid_extensions(self):
        """Test that payload paths with valid extensions are accepted."""
        for ext in [".json", ".enc", ".encrypted"]:
            path = Path(f"payload{ext}")
            validated = PathValidator.validate_payload_path(path, must_exist=False)
            assert validated.suffix.lower() == ext
    
    def test_validate_config_path_valid_extensions(self):
        """Test that config paths with valid extensions are accepted."""
        for ext in [".yaml", ".yml", ".json"]:
            path = Path(f"config{ext}")
            validated = PathValidator.validate_config_path(path, must_exist=False)
            assert validated.suffix.lower() == ext
    
    def test_sanitize_filename_removes_path_separators(self):
        """Test that filename sanitization removes path separators."""
        sanitized = PathValidator.sanitize_filename("../../../etc/passwd")
        assert "/" not in sanitized
        assert "\\" not in sanitized
        # Note: ".." gets replaced with "_" so it becomes "_.._.._etc_passwd"
        # The important thing is path separators are removed
    
    def test_sanitize_filename_removes_null_bytes(self):
        """Test that filename sanitization removes null bytes."""
        sanitized = PathValidator.sanitize_filename("file\x00.txt")
        assert "\x00" not in sanitized
    
    def test_sanitize_filename_removes_control_characters(self):
        """Test that filename sanitization removes control characters."""
        sanitized = PathValidator.sanitize_filename("file\x01\x02.txt")
        assert "\x01" not in sanitized
        assert "\x02" not in sanitized
    
    def test_sanitize_filename_limits_length(self):
        """Test that filename sanitization limits length."""
        long_name = "a" * 300 + ".txt"
        sanitized = PathValidator.sanitize_filename(long_name)
        assert len(sanitized) <= 255
        assert sanitized.endswith(".txt")  # Extension preserved
    
    def test_sanitize_filename_empty_after_sanitization(self):
        """Test that empty filenames after sanitization raise error."""
        with pytest.raises(ValueError, match="empty after sanitization"):
            PathValidator.sanitize_filename("...")
    
    def test_is_safe_path_returns_bool(self):
        """Test that is_safe_path returns boolean without raising."""
        assert PathValidator.is_safe_path(Path("safe.txt")) is True
        assert PathValidator.is_safe_path(Path("../../../etc/passwd")) is False


class TestSanitizer:
    """Test data sanitization for logging."""
    
    def test_sanitize_dict_removes_sensitive_keys(self):
        """Test that sensitive keys are redacted in dictionaries."""
        data = {
            "key": "secret123",
            "password": "mypassword",
            "name": "test",
            "token": "abc123"
        }
        sanitized = Sanitizer.sanitize_dict(data)
        
        assert sanitized["key"] == "***REDACTED***"
        assert sanitized["password"] == "***REDACTED***"
        assert sanitized["token"] == "***REDACTED***"
        assert sanitized["name"] == "test"  # Not sensitive
    
    def test_sanitize_dict_nested(self):
        """Test that nested dictionaries are sanitized."""
        data = {
            "outer": {
                "key": "secret",
                "safe": "value"
            }
        }
        sanitized = Sanitizer.sanitize_dict(data, deep=True)
        
        assert sanitized["outer"]["key"] == "***REDACTED***"
        assert sanitized["outer"]["safe"] == "value"
    
    def test_sanitize_dict_with_lists(self):
        """Test that lists in dictionaries are sanitized."""
        data = {
            "items": [
                {"key": "secret1"},
                {"key": "secret2"}
            ]
        }
        sanitized = Sanitizer.sanitize_dict(data, deep=True)
        
        assert sanitized["items"][0]["key"] == "***REDACTED***"
        assert sanitized["items"][1]["key"] == "***REDACTED***"
    
    def test_sanitize_list(self):
        """Test that lists are sanitized."""
        data = [
            {"key": "secret"},
            {"password": "pass123"}
        ]
        sanitized = Sanitizer.sanitize_list(data, deep=True)
        
        assert sanitized[0]["key"] == "***REDACTED***"
        assert sanitized[1]["password"] == "***REDACTED***"
    
    def test_sanitize_string_aggressive(self):
        """Test that aggressive string sanitization removes patterns."""
        # Base64-like pattern
        text = "Key: YWJjMTIzZGVmNDU2Z2hpNzg5amtsMTIzNDU2Nzg5MA=="
        sanitized = Sanitizer.sanitize_string(text, aggressive=True)
        assert "***REDACTED***" in sanitized
    
    def test_sanitize_exception_message(self):
        """Test that exception messages are sanitized."""
        message = "Encryption failed with key: secret123"
        sanitized = Sanitizer.sanitize_exception_message(message)
        assert "secret123" not in sanitized
        assert "***REDACTED***" in sanitized
    
    def test_sanitize_exception_message_multiple_patterns(self):
        """Test that multiple sensitive patterns are sanitized."""
        message = "Failed: key: abc123, password: xyz789, token: def456"
        sanitized = Sanitizer.sanitize_exception_message(message)
        assert "abc123" not in sanitized
        assert "xyz789" not in sanitized
        assert "def456" not in sanitized
    
    def test_sanitize_log_record(self):
        """Test that log records are sanitized."""
        record = {
            "message": "Processing with key: secret",
            "key": "secret123",
            "user": "testuser"
        }
        sanitized = Sanitizer.sanitize_log_record(record)
        
        assert "secret" not in sanitized["message"]
        assert sanitized["key"] == "***REDACTED***"
        assert sanitized["user"] == "testuser"
    
    def test_mask_key_shows_partial(self):
        """Test that key masking shows partial key."""
        key = "verylongsecretkey123"
        masked = Sanitizer.mask_key(key, visible_chars=4)
        assert masked == "very...y123"
        assert "secret" not in masked
    
    def test_mask_key_short_key(self):
        """Test that short keys are fully redacted."""
        key = "short"
        masked = Sanitizer.mask_key(key, visible_chars=4)
        assert masked == "***REDACTED***"
    
    def test_mask_key_bytes(self):
        """Test that byte keys are masked."""
        key = b"secretkey123"
        masked = Sanitizer.mask_key(key, visible_chars=4)
        assert "***REDACTED***" in masked or "..." in masked
    
    def test_is_sensitive_key_detection(self):
        """Test that sensitive keys are detected."""
        assert Sanitizer._is_sensitive_key("key")
        assert Sanitizer._is_sensitive_key("password")
        assert Sanitizer._is_sensitive_key("api_key")
        assert Sanitizer._is_sensitive_key("encryption_key")
        assert not Sanitizer._is_sensitive_key("name")
        assert not Sanitizer._is_sensitive_key("value")


class TestSecurityIntegration:
    """Integration tests for security measures."""
    
    def test_key_validation_prevents_injection(self):
        """Test that key validation prevents injection attacks."""
        malicious_keys = [
            "'; DROP TABLE users--",
            "password && rm -rf /",
            "../../../etc/passwd",
            "password\x00malicious"
        ]
        
        for key in malicious_keys:
            with pytest.raises((ValueError, TypeError)):
                InputValidator.validate_key(key)
    
    def test_path_validation_prevents_traversal(self):
        """Test that path validation prevents traversal attacks."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\Windows\\System32",
            "~/../../etc/passwd"
        ]
        
        for path in malicious_paths:
            with pytest.raises(ValueError):
                PathValidator.validate_path(Path(path))
    
    def test_sanitizer_prevents_key_leakage(self):
        """Test that sanitizer prevents key leakage in logs."""
        sensitive_data = {
            "encryption_key": "supersecret123",
            "password": "mypassword",
            "user": "testuser",
            "operation": "encrypt"
        }
        
        sanitized = Sanitizer.sanitize_dict(sensitive_data)
        
        # Sensitive data should be redacted
        assert "supersecret123" not in str(sanitized)
        assert "mypassword" not in str(sanitized)
        
        # Non-sensitive data should be preserved
        assert sanitized["user"] == "testuser"
        assert sanitized["operation"] == "encrypt"
    
    def test_end_to_end_input_validation(self):
        """Test end-to-end input validation workflow."""
        # Valid inputs should pass all validations
        key = "ValidPassword123!"
        coefficient_count = 100
        animation_speed = 1.5
        
        assert InputValidator.validate_key(key)
        assert InputValidator.validate_coefficient_count(coefficient_count)
        assert InputValidator.validate_animation_speed(animation_speed)
        
        # Invalid inputs should be rejected
        with pytest.raises(ValueError):
            InputValidator.validate_key("short")
        
        with pytest.raises(ValueError):
            InputValidator.validate_coefficient_count(5000)
        
        with pytest.raises(ValueError):
            InputValidator.validate_animation_speed(100.0)
