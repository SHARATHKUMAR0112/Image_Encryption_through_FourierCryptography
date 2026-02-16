"""
Input validation utilities for security.

This module provides comprehensive input validation to prevent injection attacks,
validate data types and ranges, and ensure all user inputs are safe.
"""

import re
from typing import Any, Dict, List, Optional, Union


class InputValidator:
    """
    Validate user inputs to prevent injection attacks and ensure data integrity.
    
    Provides validation for:
    - Encryption keys and passwords
    - Numeric ranges (coefficients, speeds, iterations)
    - String inputs (no injection patterns)
    - Configuration values
    
    Requirements:
        - 3.18.3: System shall validate all user inputs to prevent injection attacks
    """
    
    # Injection attack patterns
    SQL_INJECTION_PATTERNS = [
        re.compile(r"(\bOR\b|\bAND\b).*[=<>]", re.IGNORECASE),
        re.compile(r";\s*(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE)\s+", re.IGNORECASE),
        re.compile(r"--", re.IGNORECASE),
        re.compile(r"/\*.*\*/", re.IGNORECASE),
        re.compile(r"'\s*(OR|AND)\s*'", re.IGNORECASE),
        re.compile(r"UNION\s+SELECT", re.IGNORECASE),
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        re.compile(r"[;&|`$]"),  # Shell metacharacters
        re.compile(r"\$\(.*\)"),  # Command substitution
        re.compile(r"`.*`"),  # Backtick command substitution
        re.compile(r">\s*/dev/"),  # Device file access
        re.compile(r"\|\s*\w+"),  # Pipe to command
    ]
    
    PATH_INJECTION_PATTERNS = [
        re.compile(r"\.\./"),  # Path traversal
        re.compile(r"\.\.\\"),  # Windows path traversal
        re.compile(r"~[/\\]"),  # Home directory
        re.compile(r"^/etc/"),  # System directories
        re.compile(r"^/proc/"),
        re.compile(r"^/sys/"),
        re.compile(r"^C:\\Windows\\", re.IGNORECASE),
    ]
    
    # Valid ranges
    COEFFICIENT_COUNT_MIN = 10
    COEFFICIENT_COUNT_MAX = 1000
    ANIMATION_SPEED_MIN = 0.1
    ANIMATION_SPEED_MAX = 10.0
    KDF_ITERATIONS_MIN = 100_000
    KDF_ITERATIONS_MAX = 10_000_000
    
    @classmethod
    def validate_key(cls, key: str, min_length: int = 8) -> bool:
        """
        Validate an encryption key/password.
        
        Checks:
        - Minimum length
        - No null bytes
        - No control characters
        - No obvious injection patterns
        
        Args:
            key: Key/password to validate
            min_length: Minimum required length
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If key is invalid
            
        Example:
            >>> InputValidator.validate_key("mypassword123")
            True
            >>> InputValidator.validate_key("short")  # Raises ValueError
        """
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        
        if not key:
            raise ValueError("Key cannot be empty")
        
        if len(key) < min_length:
            raise ValueError(
                f"Key must be at least {min_length} characters long, got {len(key)}"
            )
        
        # Check for null bytes
        if "\x00" in key:
            raise ValueError("Key contains null bytes")
        
        # Check for control characters (except tab, newline, carriage return)
        for char in key:
            if ord(char) < 32 and char not in ["\t", "\n", "\r"]:
                raise ValueError("Key contains control characters")
        
        # Check for injection patterns
        if cls._contains_injection_pattern(key):
            raise ValueError("Key contains suspicious patterns")
        
        return True
    
    @classmethod
    def validate_coefficient_count(cls, count: int) -> bool:
        """
        Validate Fourier coefficient count.
        
        Args:
            count: Number of coefficients
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If count is out of range
            TypeError: If count is not an integer
        """
        if not isinstance(count, int):
            raise TypeError(f"Coefficient count must be an integer, got {type(count)}")
        
        if count < cls.COEFFICIENT_COUNT_MIN or count > cls.COEFFICIENT_COUNT_MAX:
            raise ValueError(
                f"Coefficient count must be between {cls.COEFFICIENT_COUNT_MIN} "
                f"and {cls.COEFFICIENT_COUNT_MAX}, got {count}"
            )
        
        return True
    
    @classmethod
    def validate_animation_speed(cls, speed: float) -> bool:
        """
        Validate animation speed multiplier.
        
        Args:
            speed: Animation speed multiplier
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If speed is out of range
            TypeError: If speed is not a number
        """
        if not isinstance(speed, (int, float)):
            raise TypeError(f"Animation speed must be a number, got {type(speed)}")
        
        if speed < cls.ANIMATION_SPEED_MIN or speed > cls.ANIMATION_SPEED_MAX:
            raise ValueError(
                f"Animation speed must be between {cls.ANIMATION_SPEED_MIN} "
                f"and {cls.ANIMATION_SPEED_MAX}, got {speed}"
            )
        
        return True
    
    @classmethod
    def validate_kdf_iterations(cls, iterations: int) -> bool:
        """
        Validate KDF iteration count.
        
        Args:
            iterations: Number of KDF iterations
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If iterations is out of range
            TypeError: If iterations is not an integer
        """
        if not isinstance(iterations, int):
            raise TypeError(f"KDF iterations must be an integer, got {type(iterations)}")
        
        if iterations < cls.KDF_ITERATIONS_MIN or iterations > cls.KDF_ITERATIONS_MAX:
            raise ValueError(
                f"KDF iterations must be between {cls.KDF_ITERATIONS_MIN} "
                f"and {cls.KDF_ITERATIONS_MAX}, got {iterations}"
            )
        
        return True
    
    @classmethod
    def validate_string_input(
        cls,
        value: str,
        max_length: Optional[int] = None,
        allow_empty: bool = False,
        check_injection: bool = True,
    ) -> bool:
        """
        Validate a string input for safety.
        
        Args:
            value: String to validate
            max_length: Maximum allowed length (optional)
            allow_empty: If True, allow empty strings
            check_injection: If True, check for injection patterns
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If string is invalid
            TypeError: If value is not a string
        """
        if not isinstance(value, str):
            raise TypeError(f"Value must be a string, got {type(value)}")
        
        if not allow_empty and not value:
            raise ValueError("String cannot be empty")
        
        if max_length is not None and len(value) > max_length:
            raise ValueError(
                f"String exceeds maximum length of {max_length}, got {len(value)}"
            )
        
        # Check for null bytes
        if "\x00" in value:
            raise ValueError("String contains null bytes")
        
        # Check for injection patterns
        if check_injection and cls._contains_injection_pattern(value):
            raise ValueError("String contains suspicious injection patterns")
        
        return True
    
    @classmethod
    def validate_config_dict(cls, config: Dict[str, Any]) -> bool:
        """
        Validate a configuration dictionary.
        
        Ensures all keys and values are safe and within expected ranges.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If configuration is invalid
            TypeError: If config is not a dictionary
        """
        if not isinstance(config, dict):
            raise TypeError(f"Config must be a dictionary, got {type(config)}")
        
        # Validate each key-value pair
        for key, value in config.items():
            # Validate key
            if not isinstance(key, str):
                raise TypeError(f"Config key must be string, got {type(key)}")
            
            cls.validate_string_input(key, max_length=100, check_injection=True)
            
            # Validate value based on type
            if isinstance(value, str):
                cls.validate_string_input(value, max_length=1000, check_injection=True)
            elif isinstance(value, dict):
                # Recursively validate nested dictionaries
                cls.validate_config_dict(value)
            elif isinstance(value, list):
                # Validate list items
                for item in value:
                    if isinstance(item, str):
                        cls.validate_string_input(item, max_length=1000, check_injection=True)
                    elif isinstance(item, dict):
                        cls.validate_config_dict(item)
        
        return True
    
    @classmethod
    def _contains_injection_pattern(cls, value: str) -> bool:
        """
        Check if a string contains injection attack patterns.
        
        Args:
            value: String to check
            
        Returns:
            True if injection pattern detected, False otherwise
        """
        # Check SQL injection patterns
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if pattern.search(value):
                return True
        
        # Check command injection patterns
        for pattern in cls.COMMAND_INJECTION_PATTERNS:
            if pattern.search(value):
                return True
        
        # Check path injection patterns
        for pattern in cls.PATH_INJECTION_PATTERNS:
            if pattern.search(value):
                return True
        
        return False
    
    @classmethod
    def sanitize_for_logging(cls, value: Any) -> str:
        """
        Sanitize a value for safe logging.
        
        Removes or masks sensitive patterns before logging.
        
        Args:
            value: Value to sanitize
            
        Returns:
            Sanitized string representation
        """
        from fourier_encryption.security.sanitizer import Sanitizer
        
        if isinstance(value, str):
            return Sanitizer.sanitize_string(value, aggressive=True)
        elif isinstance(value, dict):
            return str(Sanitizer.sanitize_dict(value))
        elif isinstance(value, (list, tuple)):
            return str(Sanitizer.sanitize_list(list(value)))
        else:
            return str(value)
