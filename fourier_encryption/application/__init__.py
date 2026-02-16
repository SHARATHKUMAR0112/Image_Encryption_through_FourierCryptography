"""
Application layer for Fourier-Based Image Encryption System.

This module contains the orchestration logic that coordinates the entire
encryption and decryption workflows, managing dependencies between components.
"""

from fourier_encryption.application.orchestrator import EncryptionOrchestrator

__all__ = ['EncryptionOrchestrator']
