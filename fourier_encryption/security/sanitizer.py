"""
Data sanitization utilities for security.

This module provides functions to sanitize sensitive data in logs, error messages,
and other outputs to prevent information leakage.
"""

import re
from typing import Any, Dict, List, Set, Union


class Sanitizer:
    """
    Sanitize sensitive data from logs, errors, and outputs.
    
    Prevents accidental exposure of encryption keys, passwords, tokens,
    and other sensitive information in logs and error messages.
    
    Requirements:
        - 3.18.1: System shall never log or display encryption keys
    """
    
    # Sensitive field names (case-insensitive)
    SENSITIVE_KEYS: Set[str] = {
        "key",
        "password",
        "passwd",
        "pwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "private_key",
        "privatekey",
        "auth",
        "authorization",
        "credential",
        "salt",
        "iv",
        "hmac",
        "signature",
        "session",
        "cookie",
    }
    
    # Patterns that might contain sensitive data
    SENSITIVE_PATTERNS = [
        # Base64-encoded data (potential keys/tokens)
        re.compile(r'[A-Za-z0-9+/]{32,}={0,2}'),
        # Hex-encoded data (potential keys/hashes)
        re.compile(r'[0-9a-fA-F]{32,}'),
        # JWT tokens
        re.compile(r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+'),
    ]
    
    REDACTION_TEXT = "***REDACTED***"
    
    @classmethod
    def sanitize_dict(cls, data: Dict[str, Any], deep: bool = True) -> Dict[str, Any]:
        """
        Sanitize sensitive fields in a dictionary.
        
        Recursively scans dictionary for sensitive keys and replaces their
        values with redaction text. Handles nested dictionaries and lists.
        
        Args:
            data: Dictionary to sanitize
            deep: If True, recursively sanitize nested structures
            
        Returns:
            Sanitized dictionary with sensitive values redacted
            
        Example:
            >>> Sanitizer.sanitize_dict({"key": "secret123", "name": "test"})
            {"key": "***REDACTED***", "name": "test"}
        """
        if not isinstance(data, dict):
            return data
        
        sanitized = {}
        for key, value in data.items():
            # Check if key is sensitive
            if cls._is_sensitive_key(key):
                sanitized[key] = cls.REDACTION_TEXT
            elif deep and isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized[key] = cls.sanitize_dict(value, deep=True)
            elif deep and isinstance(value, list):
                # Sanitize lists
                sanitized[key] = cls.sanitize_list(value, deep=True)
            else:
                sanitized[key] = value
        
        return sanitized
    
    @classmethod
    def sanitize_list(cls, data: List[Any], deep: bool = True) -> List[Any]:
        """
        Sanitize sensitive data in a list.
        
        Args:
            data: List to sanitize
            deep: If True, recursively sanitize nested structures
            
        Returns:
            Sanitized list
        """
        if not isinstance(data, list):
            return data
        
        sanitized = []
        for item in data:
            if isinstance(item, dict):
                sanitized.append(cls.sanitize_dict(item, deep=deep))
            elif isinstance(item, list):
                sanitized.append(cls.sanitize_list(item, deep=deep))
            else:
                sanitized.append(item)
        
        return sanitized
    
    @classmethod
    def sanitize_string(cls, text: str, aggressive: bool = False) -> str:
        """
        Sanitize sensitive patterns in a string.
        
        Removes or redacts patterns that might contain sensitive data like
        base64-encoded keys, hex-encoded hashes, or JWT tokens.
        
        Args:
            text: String to sanitize
            aggressive: If True, redact all potential sensitive patterns
            
        Returns:
            Sanitized string
            
        Example:
            >>> Sanitizer.sanitize_string("Key: abc123def456...", aggressive=True)
            "Key: ***REDACTED***"
        """
        if not isinstance(text, str):
            return text
        
        sanitized = text
        
        if aggressive:
            # Redact all patterns that might be sensitive
            for pattern in cls.SENSITIVE_PATTERNS:
                sanitized = pattern.sub(cls.REDACTION_TEXT, sanitized)
        
        return sanitized
    
    @classmethod
    def sanitize_exception_message(cls, message: str) -> str:
        """
        Sanitize exception messages to remove sensitive data.
        
        Ensures that exception messages don't leak encryption keys,
        passwords, or other sensitive information.
        
        Args:
            message: Exception message to sanitize
            
        Returns:
            Sanitized exception message
            
        Example:
            >>> Sanitizer.sanitize_exception_message("Failed with key: secret123")
            "Failed with key: ***REDACTED***"
        """
        # Replace common patterns that might expose keys
        sanitized = message
        
        # Pattern: "key: <value>" or "password: <value>"
        for key_name in ["key", "password", "secret", "token"]:
            pattern = re.compile(
                rf'{key_name}[:\s=]+["\']?([^\s"\']+)["\']?',
                re.IGNORECASE
            )
            sanitized = pattern.sub(
                f'{key_name}: {cls.REDACTION_TEXT}',
                sanitized
            )
        
        return sanitized
    
    @classmethod
    def sanitize_log_record(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize a log record before output.
        
        Ensures that log records don't contain sensitive information.
        Works with structured logging formats.
        
        Args:
            record: Log record dictionary
            
        Returns:
            Sanitized log record
        """
        sanitized = cls.sanitize_dict(record, deep=True)
        
        # Also sanitize the message field if present
        if "message" in sanitized and isinstance(sanitized["message"], str):
            sanitized["message"] = cls.sanitize_exception_message(sanitized["message"])
        
        if "msg" in sanitized and isinstance(sanitized["msg"], str):
            sanitized["msg"] = cls.sanitize_exception_message(sanitized["msg"])
        
        return sanitized
    
    @classmethod
    def _is_sensitive_key(cls, key: str) -> bool:
        """
        Check if a key name indicates sensitive data.
        
        Args:
            key: Key name to check
            
        Returns:
            True if key is sensitive, False otherwise
        """
        if not isinstance(key, str):
            return False
        
        key_lower = key.lower()
        
        # Check exact matches
        if key_lower in cls.SENSITIVE_KEYS:
            return True
        
        # Check if any sensitive keyword is contained in the key
        for sensitive in cls.SENSITIVE_KEYS:
            if sensitive in key_lower:
                return True
        
        return False
    
    @classmethod
    def mask_key(cls, key: Union[str, bytes], visible_chars: int = 4) -> str:
        """
        Mask a key showing only first/last few characters.
        
        Useful for debugging where you need to identify which key is being
        used without exposing the full key value.
        
        Args:
            key: Key to mask (string or bytes)
            visible_chars: Number of characters to show at start and end
            
        Returns:
            Masked key string
            
        Example:
            >>> Sanitizer.mask_key("verylongsecretkey123", visible_chars=4)
            "very...y123"
        """
        if isinstance(key, bytes):
            key = key.hex()
        
        if not isinstance(key, str):
            return cls.REDACTION_TEXT
        
        if len(key) <= visible_chars * 2:
            return cls.REDACTION_TEXT
        
        return f"{key[:visible_chars]}...{key[-visible_chars:]}"
