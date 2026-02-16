"""
Main entry point for the Fourier-Based Image Encryption System.

This module provides the primary interface for running the encryption system.
"""

from fourier_encryption.cli.commands import app


def main() -> None:
    """Main application entry point - launches the CLI."""
    app()


if __name__ == "__main__":
    main()
