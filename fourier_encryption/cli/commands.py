"""
Command-line interface for Fourier-Based Image Encryption System.

This module provides CLI commands for encrypting and decrypting images using
the Fourier-based encryption system with epicycle visualization.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table

from fourier_encryption.application.orchestrator import EncryptionOrchestrator
from fourier_encryption.config.logging_config import setup_logging, get_logger
from fourier_encryption.config.settings import (
    SystemConfig,
    PreprocessConfig,
    EncryptionConfig,
)
from fourier_encryption.core.contour_extractor import ContourExtractor
from fourier_encryption.core.edge_detector import CannyEdgeDetector
from fourier_encryption.core.fourier_transformer import FourierTransformer
from fourier_encryption.core.image_processor import OpenCVImageProcessor
from fourier_encryption.encryption.aes_encryptor import AES256Encryptor
from fourier_encryption.encryption.key_manager import KeyManager
from fourier_encryption.models.data_models import EncryptedPayload
from fourier_encryption.models.exceptions import (
    FourierEncryptionError,
    ConfigurationError,
)
from fourier_encryption.transmission.serializer import CoefficientSerializer
from fourier_encryption.visualization.live_renderer import LiveRenderer
from fourier_encryption.core.epicycle_engine import EpicycleEngine
from fourier_encryption.security.input_validator import InputValidator
from fourier_encryption.security.path_validator import PathValidator


# Initialize Typer app
app = typer.Typer(
    name="fourier-encrypt",
    help="Fourier-Based Image Encryption System with AI Integration",
    add_completion=False,
)

# Initialize console for rich output
console = Console()

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def create_orchestrator(config: Optional[SystemConfig] = None) -> EncryptionOrchestrator:
    """
    Create and configure the encryption orchestrator with all dependencies.
    
    Args:
        config: Optional system configuration. If None, uses defaults.
        
    Returns:
        Configured EncryptionOrchestrator instance
    """
    # Create core components
    image_processor = OpenCVImageProcessor()
    edge_detector = CannyEdgeDetector()
    contour_extractor = ContourExtractor()
    fourier_transformer = FourierTransformer()
    encryptor = AES256Encryptor()
    serializer = CoefficientSerializer()
    key_manager = KeyManager()
    
    # Create orchestrator
    orchestrator = EncryptionOrchestrator(
        image_processor=image_processor,
        edge_detector=edge_detector,
        contour_extractor=contour_extractor,
        fourier_transformer=fourier_transformer,
        encryptor=encryptor,
        serializer=serializer,
        key_manager=key_manager,
        optimizer=None,  # AI components optional
        anomaly_detector=None,
    )
    
    return orchestrator


def load_config(config_path: Optional[Path]) -> SystemConfig:
    """
    Load system configuration from file or use defaults.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        SystemConfig instance
        
    Raises:
        ConfigurationError: If config file is invalid
    """
    if config_path:
        try:
            return SystemConfig.from_file(config_path)
        except ConfigurationError as e:
            console.print(f"[red]Configuration error: {e}[/red]")
            raise typer.Exit(code=1)
    else:
        # Use default configuration
        return SystemConfig()


def save_encrypted_payload(payload: EncryptedPayload, output_path: Path) -> None:
    """
    Save encrypted payload to file in JSON format.
    
    Args:
        payload: EncryptedPayload to save
        output_path: Path to output file
    """
    # Convert payload to JSON-serializable format
    payload_dict = {
        "ciphertext": payload.ciphertext.hex(),
        "iv": payload.iv.hex(),
        "hmac": payload.hmac.hex(),
        "metadata": payload.metadata,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload_dict, f, indent=2)


def load_encrypted_payload(input_path: Path) -> EncryptedPayload:
    """
    Load encrypted payload from JSON file.
    
    Args:
        input_path: Path to encrypted payload file
        
    Returns:
        EncryptedPayload instance
        
    Raises:
        ValueError: If file format is invalid
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        payload_dict = json.load(f)
    
    # Convert hex strings back to bytes
    return EncryptedPayload(
        ciphertext=bytes.fromhex(payload_dict["ciphertext"]),
        iv=bytes.fromhex(payload_dict["iv"]),
        hmac=bytes.fromhex(payload_dict["hmac"]),
        metadata=payload_dict["metadata"],
    )


@app.command()
def encrypt(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to input image file (PNG, JPG, BMP)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to output encrypted file (JSON format)",
        file_okay=True,
        dir_okay=False,
        writable=True,
    ),
    key: str = typer.Option(
        ...,
        "--key",
        "-k",
        help="Encryption password/key (minimum 8 characters)",
        prompt=True,
        hide_input=True,
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (YAML or JSON)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    coefficients: Optional[int] = typer.Option(
        None,
        "--coefficients",
        "-n",
        help="Number of Fourier coefficients (10-1000, default: auto)",
        min=10,
        max=1000,
    ),
) -> None:
    """
    Encrypt an image using Fourier-based encryption.
    
    This command processes an image through edge detection, Fourier transform,
    and AES-256 encryption to produce a secure encrypted payload.
    
    Example:
        fourier-encrypt encrypt -i image.png -o encrypted.json -k mypassword
    """
    console.print(Panel.fit(
        "[bold cyan]Fourier-Based Image Encryption[/bold cyan]",
        border_style="cyan"
    ))
    
    try:
        # Validate input path
        try:
            validated_input = PathValidator.validate_image_path(input, must_exist=True)
        except (ValueError, FileNotFoundError) as e:
            console.print(f"[red]Invalid input path: {e}[/red]")
            raise typer.Exit(code=1)
        
        # Validate output path
        try:
            validated_output = PathValidator.validate_payload_path(output, must_exist=False)
        except ValueError as e:
            console.print(f"[red]Invalid output path: {e}[/red]")
            raise typer.Exit(code=1)
        
        # Validate encryption key
        try:
            InputValidator.validate_key(key, min_length=8)
        except (ValueError, TypeError) as e:
            console.print(f"[red]Invalid encryption key: {e}[/red]")
            raise typer.Exit(code=1)
        
        # Validate coefficient count if provided
        if coefficients is not None:
            try:
                InputValidator.validate_coefficient_count(coefficients)
            except (ValueError, TypeError) as e:
                console.print(f"[red]Invalid coefficient count: {e}[/red]")
                raise typer.Exit(code=1)
        
        # Validate config path if provided
        if config is not None:
            try:
                validated_config = PathValidator.validate_config_path(config, must_exist=True)
            except (ValueError, FileNotFoundError) as e:
                console.print(f"[red]Invalid config path: {e}[/red]")
                raise typer.Exit(code=1)
        
        # Load configuration
        system_config = load_config(config)
        
        # Override coefficient count if specified
        if coefficients:
            system_config.encryption.num_coefficients = coefficients
        
        # Create orchestrator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Initializing encryption system...", total=None)
            orchestrator = create_orchestrator(system_config)
            progress.update(task, completed=True)
        
        # Display encryption settings
        settings_table = Table(title="Encryption Settings", show_header=False)
        settings_table.add_column("Setting", style="cyan")
        settings_table.add_column("Value", style="green")
        settings_table.add_row("Input Image", str(input))
        settings_table.add_row("Output File", str(output))
        settings_table.add_row(
            "Coefficients",
            str(system_config.encryption.num_coefficients or "Auto-optimize")
        )
        settings_table.add_row(
            "KDF Iterations",
            f"{system_config.encryption.kdf_iterations:,}"
        )
        console.print(settings_table)
        console.print()
        
        # Encrypt image with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Encrypting image...", total=8)
            
            progress.update(task, advance=1, description="[cyan]Loading image...")
            progress.update(task, advance=1, description="[cyan]Detecting edges...")
            progress.update(task, advance=1, description="[cyan]Extracting contours...")
            progress.update(task, advance=1, description="[cyan]Computing Fourier transform...")
            
            # Perform encryption
            encrypted_payload = orchestrator.encrypt_image(
                validated_input,
                key,
                system_config.preprocessing,
                system_config.encryption,
            )
            
            progress.update(task, advance=4, description="[cyan]Finalizing encryption...")
        
        # Save encrypted payload
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Saving encrypted file...", total=None)
            save_encrypted_payload(encrypted_payload, validated_output)
            progress.update(task, completed=True)
        
        # Display success message
        console.print()
        console.print(Panel.fit(
            f"[bold green]✓ Encryption successful![/bold green]\n\n"
            f"Encrypted file saved to: [cyan]{validated_output}[/cyan]\n"
            f"Payload size: [yellow]{len(encrypted_payload.ciphertext):,}[/yellow] bytes\n"
            f"Coefficients: [yellow]{encrypted_payload.metadata.get('original_contour_length', 'N/A')}[/yellow]",
            border_style="green"
        ))
        
        logger.info(
            "Encryption completed successfully",
            extra={
                "input": str(validated_input),
                "output": str(validated_output),
                "payload_size": len(encrypted_payload.ciphertext),
            }
        )
        
    except FourierEncryptionError as e:
        console.print(f"\n[bold red]✗ Encryption failed:[/bold red] {e}")
        logger.error(f"Encryption failed: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]✗ Unexpected error:[/bold red] {e}")
        logger.error(f"Unexpected error during encryption: {e}")
        raise typer.Exit(code=1)


@app.command()
def decrypt(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to encrypted file (JSON format)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to output reconstructed contour file (NPY format)",
        file_okay=True,
        dir_okay=False,
        writable=True,
    ),
    key: str = typer.Option(
        ...,
        "--key",
        "-k",
        help="Decryption password/key",
        prompt=True,
        hide_input=True,
    ),
    visualize: bool = typer.Option(
        False,
        "--visualize",
        "-v",
        help="Enable live epicycle animation during decryption",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (YAML or JSON)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    speed: float = typer.Option(
        4.0,
        "--speed",
        "-s",
        help="Animation speed multiplier (0.1-10.0, only with --visualize)",
        min=0.1,
        max=10.0,
    ),
) -> None:
    """
    Decrypt an encrypted image and optionally visualize the reconstruction.
    
    This command decrypts the encrypted payload, reconstructs the Fourier
    coefficients, and optionally displays a live epicycle animation.
    
    Example:
        fourier-encrypt decrypt -i encrypted.json -o contour.npy -k mypassword
        fourier-encrypt decrypt -i encrypted.json -o contour.npy -k mypassword --visualize
    """
    console.print(Panel.fit(
        "[bold cyan]Fourier-Based Image Decryption[/bold cyan]",
        border_style="cyan"
    ))
    
    try:
        # Validate input path
        try:
            validated_input = PathValidator.validate_payload_path(input, must_exist=True)
        except (ValueError, FileNotFoundError) as e:
            console.print(f"[red]Invalid input path: {e}[/red]")
            raise typer.Exit(code=1)
        
        # Validate output path (allow .npy extension for contour output)
        try:
            validated_output = PathValidator.validate_path(output, must_exist=False)
        except ValueError as e:
            console.print(f"[red]Invalid output path: {e}[/red]")
            raise typer.Exit(code=1)
        
        # Validate decryption key
        try:
            InputValidator.validate_key(key, min_length=8)
        except (ValueError, TypeError) as e:
            console.print(f"[red]Invalid decryption key: {e}[/red]")
            raise typer.Exit(code=1)
        
        # Validate animation speed if visualization enabled
        if visualize:
            try:
                InputValidator.validate_animation_speed(speed)
            except (ValueError, TypeError) as e:
                console.print(f"[red]Invalid animation speed: {e}[/red]")
                raise typer.Exit(code=1)
        
        # Validate config path if provided
        if config is not None:
            try:
                validated_config = PathValidator.validate_config_path(config, must_exist=True)
            except (ValueError, FileNotFoundError) as e:
                console.print(f"[red]Invalid config path: {e}[/red]")
                raise typer.Exit(code=1)
        
        # Load configuration
        system_config = load_config(config)
        
        # Create orchestrator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Initializing decryption system...", total=None)
            orchestrator = create_orchestrator(system_config)
            progress.update(task, completed=True)
        
        # Display decryption settings
        settings_table = Table(title="Decryption Settings", show_header=False)
        settings_table.add_column("Setting", style="cyan")
        settings_table.add_column("Value", style="green")
        settings_table.add_row("Input File", str(input))
        settings_table.add_row("Output File", str(output))
        settings_table.add_row("Visualization", "Enabled" if visualize else "Disabled")
        if visualize:
            settings_table.add_row("Animation Speed", f"{speed}x")
        console.print(settings_table)
        console.print()
        
        # Load encrypted payload
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Loading encrypted file...", total=None)
            encrypted_payload = load_encrypted_payload(validated_input)
            progress.update(task, completed=True)
        
        # Decrypt with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Decrypting...", total=4)
            
            progress.update(task, advance=1, description="[cyan]Verifying HMAC...")
            progress.update(task, advance=1, description="[cyan]Decrypting payload...")
            progress.update(task, advance=1, description="[cyan]Deserializing coefficients...")
            
            # Perform decryption
            reconstructed_points = orchestrator.decrypt_image(
                encrypted_payload,
                key,
                visualize=False,  # We'll handle visualization separately
            )
            
            progress.update(task, advance=1, description="[cyan]Reconstruction complete")
        
        # Save reconstructed contour
        import numpy as np
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Saving reconstructed contour...", total=None)
            np.save(validated_output, reconstructed_points)
            progress.update(task, completed=True)
        
        # Display success message
        console.print()
        console.print(Panel.fit(
            f"[bold green]✓ Decryption successful![/bold green]\n\n"
            f"Reconstructed contour saved to: [cyan]{validated_output}[/cyan]\n"
            f"Points reconstructed: [yellow]{len(reconstructed_points):,}[/yellow]",
            border_style="green"
        ))
        
        # Show visualization if requested
        if visualize:
            console.print()
            console.print("[cyan]Starting epicycle animation...[/cyan]")
            console.print("[dim]Close the animation window to continue.[/dim]")
            
            try:
                # Deserialize coefficients for visualization
                from fourier_encryption.transmission.serializer import CoefficientSerializer
                serializer = CoefficientSerializer()
                
                # Decrypt again to get coefficients
                from fourier_encryption.encryption.aes_encryptor import AES256Encryptor
                encryptor = AES256Encryptor()
                salt = bytes.fromhex(encrypted_payload.metadata["salt"])
                derived_key = encryptor.derive_key(key, salt)
                decrypted_data = encryptor.decrypt(encrypted_payload, derived_key)
                coefficients, _ = serializer.deserialize(decrypted_data)
                
                # Create epicycle engine and renderer
                engine = EpicycleEngine(coefficients)
                renderer = LiveRenderer(backend="matplotlib")
                
                # Animate
                fps = 30
                num_frames = int(300 / speed)  # Adjust frames based on speed
                renderer.animate(engine, fps=fps, num_frames=num_frames, speed=speed)
                
                console.print("[green]Animation complete![/green]")
                
            except Exception as e:
                console.print(f"[yellow]Warning: Visualization failed: {e}[/yellow]")
                logger.warning(f"Visualization failed: {e}")
        
        logger.info(
            "Decryption completed successfully",
            extra={
                "input": str(validated_input),
                "output": str(validated_output),
                "points": len(reconstructed_points),
            }
        )
        
    except FourierEncryptionError as e:
        console.print(f"\n[bold red]✗ Decryption failed:[/bold red] {e}")
        logger.error(f"Decryption failed: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]✗ Unexpected error:[/bold red] {e}")
        logger.error(f"Unexpected error during decryption: {e}")
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Display version information."""
    console.print(Panel.fit(
        "[bold cyan]Fourier-Based Image Encryption System[/bold cyan]\n\n"
        "Version: [yellow]1.0.0[/yellow]\n"
        "Python: [yellow]3.9+[/yellow]\n\n"
        "A secure image encryption system using Fourier Series decomposition\n"
        "and epicycle-based sketch reconstruction with AI integration.",
        border_style="cyan"
    ))


@app.callback()
def main() -> None:
    """
    Fourier-Based Image Encryption System with AI Integration.
    
    Encrypt and decrypt images using Fourier Series decomposition with
    epicycle visualization and AI-enhanced processing.
    """
    pass


if __name__ == "__main__":
    app()
