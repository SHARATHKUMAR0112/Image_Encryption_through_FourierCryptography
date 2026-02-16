"""
Path validation and traversal prevention utilities.

This module provides secure path handling to prevent path traversal attacks
and ensure file operations stay within allowed directories.
"""

import os
from pathlib import Path
from typing import Optional, Set


class PathValidator:
    """
    Validate and sanitize file paths to prevent path traversal attacks.
    
    Prevents malicious paths like:
    - ../../../etc/passwd
    - /etc/passwd
    - C:\\Windows\\System32\\config\\sam
    - Symlink attacks
    
    Requirements:
        - 3.18.3: System shall validate all user inputs to prevent injection attacks
    """
    
    # Dangerous path components
    DANGEROUS_PATTERNS: Set[str] = {
        "..",
        "~",
        "//",
        "\\\\",
    }
    
    # Allowed file extensions for images
    ALLOWED_IMAGE_EXTENSIONS: Set[str] = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
    }
    
    # Allowed file extensions for encrypted payloads
    ALLOWED_PAYLOAD_EXTENSIONS: Set[str] = {
        ".json",
        ".enc",
        ".encrypted",
    }
    
    # Allowed file extensions for configuration
    ALLOWED_CONFIG_EXTENSIONS: Set[str] = {
        ".yaml",
        ".yml",
        ".json",
    }
    
    @classmethod
    def validate_path(
        cls,
        path: Path,
        base_dir: Optional[Path] = None,
        must_exist: bool = False,
        allow_absolute: bool = False,
    ) -> Path:
        """
        Validate a file path for security.
        
        Checks for:
        - Path traversal attempts (../)
        - Absolute paths (if not allowed)
        - Symlink attacks
        - Path stays within base directory
        
        Args:
            path: Path to validate
            base_dir: Base directory that path must be within (optional)
            must_exist: If True, path must exist
            allow_absolute: If True, allow absolute paths
            
        Returns:
            Validated and resolved Path object
            
        Raises:
            ValueError: If path is invalid or dangerous
            FileNotFoundError: If must_exist=True and path doesn't exist
            
        Example:
            >>> PathValidator.validate_path(Path("image.png"), base_dir=Path("/data"))
            Path("/data/image.png")
            >>> PathValidator.validate_path(Path("../etc/passwd"))  # Raises ValueError
        """
        if not isinstance(path, Path):
            path = Path(path)
        
        # Check for dangerous patterns in string representation
        path_str = str(path)
        for pattern in cls.DANGEROUS_PATTERNS:
            if pattern in path_str:
                raise ValueError(
                    f"Path contains dangerous pattern '{pattern}': {path_str}"
                )
        
        # Check if absolute path is allowed
        if path.is_absolute() and not allow_absolute:
            raise ValueError(
                f"Absolute paths not allowed: {path_str}"
            )
        
        # Resolve path to absolute form
        try:
            resolved_path = path.resolve(strict=must_exist)
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Cannot resolve path: {path_str}") from e
        
        # Check if path exists when required
        if must_exist and not resolved_path.exists():
            raise FileNotFoundError(f"Path does not exist: {path_str}")
        
        # Check if path is within base directory
        if base_dir is not None:
            base_dir_resolved = base_dir.resolve()
            try:
                # Check if resolved_path is relative to base_dir
                resolved_path.relative_to(base_dir_resolved)
            except ValueError:
                raise ValueError(
                    f"Path is outside base directory: {path_str} "
                    f"(base: {base_dir_resolved})"
                )
        
        # Check for symlink attacks
        if resolved_path.is_symlink():
            # Resolve symlink and check again
            real_path = resolved_path.readlink()
            if base_dir is not None:
                try:
                    real_path.relative_to(base_dir.resolve())
                except ValueError:
                    raise ValueError(
                        f"Symlink points outside base directory: {path_str}"
                    )
        
        return resolved_path
    
    @classmethod
    def validate_image_path(
        cls,
        path: Path,
        base_dir: Optional[Path] = None,
        must_exist: bool = True,
    ) -> Path:
        """
        Validate an image file path.
        
        Ensures path is safe and has an allowed image extension.
        
        Args:
            path: Image path to validate
            base_dir: Base directory that path must be within
            must_exist: If True, file must exist
            
        Returns:
            Validated Path object
            
        Raises:
            ValueError: If path is invalid or has wrong extension
        """
        validated_path = cls.validate_path(
            path,
            base_dir=base_dir,
            must_exist=must_exist,
            allow_absolute=False,
        )
        
        # Check file extension
        if validated_path.suffix.lower() not in cls.ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Invalid image file extension: {validated_path.suffix}. "
                f"Allowed: {cls.ALLOWED_IMAGE_EXTENSIONS}"
            )
        
        return validated_path
    
    @classmethod
    def validate_payload_path(
        cls,
        path: Path,
        base_dir: Optional[Path] = None,
        must_exist: bool = False,
    ) -> Path:
        """
        Validate an encrypted payload file path.
        
        Args:
            path: Payload path to validate
            base_dir: Base directory that path must be within
            must_exist: If True, file must exist
            
        Returns:
            Validated Path object
            
        Raises:
            ValueError: If path is invalid or has wrong extension
        """
        validated_path = cls.validate_path(
            path,
            base_dir=base_dir,
            must_exist=must_exist,
            allow_absolute=False,
        )
        
        # Check file extension
        if validated_path.suffix.lower() not in cls.ALLOWED_PAYLOAD_EXTENSIONS:
            raise ValueError(
                f"Invalid payload file extension: {validated_path.suffix}. "
                f"Allowed: {cls.ALLOWED_PAYLOAD_EXTENSIONS}"
            )
        
        return validated_path
    
    @classmethod
    def validate_config_path(
        cls,
        path: Path,
        base_dir: Optional[Path] = None,
        must_exist: bool = True,
    ) -> Path:
        """
        Validate a configuration file path.
        
        Args:
            path: Config path to validate
            base_dir: Base directory that path must be within
            must_exist: If True, file must exist
            
        Returns:
            Validated Path object
            
        Raises:
            ValueError: If path is invalid or has wrong extension
        """
        validated_path = cls.validate_path(
            path,
            base_dir=base_dir,
            must_exist=must_exist,
            allow_absolute=False,
        )
        
        # Check file extension
        if validated_path.suffix.lower() not in cls.ALLOWED_CONFIG_EXTENSIONS:
            raise ValueError(
                f"Invalid config file extension: {validated_path.suffix}. "
                f"Allowed: {cls.ALLOWED_CONFIG_EXTENSIONS}"
            )
        
        return validated_path
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Sanitize a filename to remove dangerous characters.
        
        Removes or replaces characters that could be used for attacks:
        - Path separators (/, \\)
        - Null bytes
        - Control characters
        - Leading/trailing dots and spaces
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Sanitized filename
            
        Example:
            >>> PathValidator.sanitize_filename("../../../etc/passwd")
            "etc_passwd"
            >>> PathValidator.sanitize_filename("file\x00.txt")
            "file.txt"
        """
        if not isinstance(filename, str):
            raise TypeError("Filename must be a string")
        
        # Remove null bytes
        sanitized = filename.replace("\x00", "")
        
        # Remove path separators
        sanitized = sanitized.replace("/", "_").replace("\\", "_")
        
        # Remove control characters
        sanitized = "".join(char for char in sanitized if ord(char) >= 32)
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip(". ")
        
        # Ensure filename is not empty
        if not sanitized:
            raise ValueError("Filename is empty after sanitization")
        
        # Limit length
        max_length = 255
        if len(sanitized) > max_length:
            # Keep extension if present
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:max_length - len(ext)] + ext
        
        return sanitized
    
    @classmethod
    def is_safe_path(cls, path: Path, base_dir: Optional[Path] = None) -> bool:
        """
        Check if a path is safe without raising exceptions.
        
        Args:
            path: Path to check
            base_dir: Base directory that path must be within
            
        Returns:
            True if path is safe, False otherwise
        """
        try:
            cls.validate_path(path, base_dir=base_dir, must_exist=False)
            return True
        except (ValueError, FileNotFoundError):
            return False
