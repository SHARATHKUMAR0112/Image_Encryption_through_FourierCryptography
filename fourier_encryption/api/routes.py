"""
REST API routes for Fourier-Based Image Encryption System.

This module implements FastAPI endpoints for encryption, decryption, and
visualization operations with rate limiting and authentication middleware.
"""

import base64
import io
import logging
import time
from pathlib import Path
from typing import Dict, Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

from fourier_encryption.application.orchestrator import EncryptionOrchestrator
from fourier_encryption.config.settings import (
    EncryptionConfig,
    PreprocessConfig,
    SystemConfig,
)
from fourier_encryption.core.edge_detector import CannyEdgeDetector
from fourier_encryption.core.contour_extractor import ContourExtractor
from fourier_encryption.core.epicycle_engine import EpicycleEngine
from fourier_encryption.core.fourier_transformer import FourierTransformer
from fourier_encryption.core.image_processor import OpenCVImageProcessor
from fourier_encryption.encryption.aes_encryptor import AES256Encryptor
from fourier_encryption.encryption.key_manager import KeyManager
from fourier_encryption.models.data_models import EncryptedPayload
from fourier_encryption.models.exceptions import (
    EncryptionError,
    DecryptionError,
    ImageProcessingError,
)
from fourier_encryption.transmission.serializer import CoefficientSerializer
from fourier_encryption.security.input_validator import InputValidator
from fourier_encryption.security.path_validator import PathValidator
from fourier_encryption.security.sanitizer import Sanitizer


logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fourier Image Encryption API",
    description="Industrial-grade image encryption using Fourier Series and epicycles",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP Basic Authentication
security = HTTPBasic()

# Rate limiting storage (in-memory, use Redis for production)
rate_limit_storage: Dict[str, list] = {}

# Configuration for rate limiting
RATE_LIMIT_REQUESTS = 10  # Maximum requests
RATE_LIMIT_WINDOW = 60  # Time window in seconds

# Authentication credentials (use environment variables in production)
VALID_USERNAME = "admin"
VALID_PASSWORD = "secure_password_123"


# Middleware: Rate Limiting
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Rate limiting middleware to prevent abuse.
    
    Limits requests to RATE_LIMIT_REQUESTS per RATE_LIMIT_WINDOW seconds
    per client IP address.
    """
    client_ip = request.client.host
    current_time = time.time()
    
    # Initialize storage for this IP if not exists
    if client_ip not in rate_limit_storage:
        rate_limit_storage[client_ip] = []
    
    # Remove old requests outside the time window
    rate_limit_storage[client_ip] = [
        req_time for req_time in rate_limit_storage[client_ip]
        if current_time - req_time < RATE_LIMIT_WINDOW
    ]
    
    # Check if rate limit exceeded
    if len(rate_limit_storage[client_ip]) >= RATE_LIMIT_REQUESTS:
        logger.warning(
            f"Rate limit exceeded for IP: {client_ip}",
            extra={"ip": client_ip, "requests": len(rate_limit_storage[client_ip])}
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "message": f"Maximum {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds",
                "retry_after": RATE_LIMIT_WINDOW
            }
        )
    
    # Add current request timestamp
    rate_limit_storage[client_ip].append(current_time)
    
    # Process request
    response = await call_next(request)
    return response


# Dependency: Authentication
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Verify HTTP Basic Authentication credentials.
    
    Uses constant-time comparison to prevent timing attacks.
    
    Args:
        credentials: HTTP Basic Auth credentials from request
        
    Returns:
        Username if authentication successful
        
    Raises:
        HTTPException: If authentication fails
    """
    correct_username = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        VALID_USERNAME.encode("utf-8")
    )
    correct_password = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        VALID_PASSWORD.encode("utf-8")
    )
    
    if not (correct_username and correct_password):
        logger.warning(
            f"Authentication failed for user: {credentials.username}",
            extra={"username": credentials.username}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return credentials.username


# Dependency: Get orchestrator instance
def get_orchestrator() -> EncryptionOrchestrator:
    """
    Create and return an EncryptionOrchestrator instance.
    
    This dependency injection pattern allows for easy testing and
    configuration management.
    
    Returns:
        Configured EncryptionOrchestrator instance
    """
    # Initialize components
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


@app.get("/")
async def root():
    """
    Root endpoint providing API information.
    
    Returns:
        API metadata and available endpoints
    """
    return {
        "name": "Fourier Image Encryption API",
        "version": "1.0.0",
        "description": "Industrial-grade image encryption using Fourier Series",
        "endpoints": {
            "encrypt": "/encrypt",
            "decrypt": "/decrypt",
            "visualize": "/visualize",
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns:
        Service health status
    """
    return {
        "status": "healthy",
        "timestamp": time.time()
    }


@app.post("/encrypt")
async def encrypt_image(
    image: UploadFile = File(..., description="Image file to encrypt (PNG, JPG, BMP)"),
    key: str = Form(..., description="Encryption password/key"),
    num_coefficients: Optional[int] = Form(None, description="Number of Fourier coefficients (10-1000)"),
    username: str = Depends(authenticate),
    orchestrator: EncryptionOrchestrator = Depends(get_orchestrator),
):
    """
    Encrypt an image using Fourier coefficient encryption.
    
    This endpoint accepts an image file and encryption key, processes the image
    through the complete encryption pipeline, and returns the encrypted payload.
    
    Args:
        image: Uploaded image file
        key: Encryption password
        num_coefficients: Optional number of Fourier coefficients (default: auto)
        username: Authenticated username (from dependency)
        orchestrator: Encryption orchestrator instance (from dependency)
        
    Returns:
        JSON response containing encrypted payload and metadata
        
    Raises:
        HTTPException: If encryption fails or validation errors occur
    """
    logger.info(
        f"Encryption request received from user: {username}",
        extra={"filename": Sanitizer.sanitize_string(image.filename or "unknown"), "user": username}
    )
    
    try:
        # Validate image file
        if not image.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided"
            )
        
        # Sanitize filename to prevent path traversal
        try:
            safe_filename = PathValidator.sanitize_filename(image.filename)
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid filename: {str(e)}"
            )
        
        # Check file extension
        allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        file_ext = Path(safe_filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format. Allowed: {allowed_extensions}"
            )
        
        # Validate encryption key
        try:
            InputValidator.validate_key(key, min_length=8)
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid encryption key: {str(e)}"
            )
        
        # Validate num_coefficients if provided
        if num_coefficients is not None:
            try:
                InputValidator.validate_coefficient_count(num_coefficients)
            except (ValueError, TypeError) as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid coefficient count: {str(e)}"
                )
        
        # Read image data
        image_data = await image.read()
        
        # Validate image data size (prevent DoS)
        max_size = 50 * 1024 * 1024  # 50 MB
        if len(image_data) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image too large. Maximum size: {max_size} bytes"
            )
        
        # Save to temporary file with safe filename
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(image_data)
        
        try:
            # Configure encryption
            preprocess_config = PreprocessConfig()
            encryption_config = EncryptionConfig(
                num_coefficients=num_coefficients,
                use_ai_edge_detection=False,  # Disable AI for API
                use_ai_optimization=False,
                use_anomaly_detection=False,
            )
            
            # Execute encryption pipeline
            encrypted_payload = orchestrator.encrypt_image(
                image_path=temp_path,
                key=key,
                preprocess_config=preprocess_config,
                encryption_config=encryption_config,
            )
            
            # Prepare response
            response_data = {
                "status": "success",
                "message": "Image encrypted successfully",
                "encrypted_data": {
                    "ciphertext": base64.b64encode(encrypted_payload.ciphertext).decode("utf-8"),
                    "iv": base64.b64encode(encrypted_payload.iv).decode("utf-8"),
                    "hmac": base64.b64encode(encrypted_payload.hmac).decode("utf-8"),
                    "metadata": encrypted_payload.metadata,
                },
                "size_bytes": len(encrypted_payload.ciphertext),
            }
            
            logger.info(
                f"Encryption successful for user: {username}",
                extra={
                    "filename": safe_filename,
                    "size": len(encrypted_payload.ciphertext),
                    "user": username
                }
            )
            
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=response_data
            )
            
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
    
    except ImageProcessingError as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image processing failed: {str(e)}"
        )
    
    except EncryptionError as e:
        logger.error(f"Encryption error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Encryption failed: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error during encryption: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/decrypt")
async def decrypt_image(
    ciphertext: str = Form(..., description="Base64-encoded ciphertext"),
    iv: str = Form(..., description="Base64-encoded initialization vector"),
    hmac: str = Form(..., description="Base64-encoded HMAC"),
    metadata: str = Form(..., description="JSON-encoded metadata"),
    key: str = Form(..., description="Decryption password/key"),
    username: str = Depends(authenticate),
    orchestrator: EncryptionOrchestrator = Depends(get_orchestrator),
):
    """
    Decrypt an encrypted image payload.
    
    This endpoint accepts an encrypted payload and decryption key, processes
    through the decryption pipeline, and returns the reconstructed contour data.
    
    Args:
        ciphertext: Base64-encoded encrypted data
        iv: Base64-encoded initialization vector
        hmac: Base64-encoded HMAC authentication tag
        metadata: JSON-encoded metadata dictionary
        key: Decryption password
        username: Authenticated username (from dependency)
        orchestrator: Encryption orchestrator instance (from dependency)
        
    Returns:
        JSON response containing reconstructed contour points
        
    Raises:
        HTTPException: If decryption fails or validation errors occur
    """
    logger.info(
        f"Decryption request received from user: {username}",
        extra={"user": username}
    )
    
    try:
        import json
        
        # Validate decryption key
        try:
            InputValidator.validate_key(key, min_length=8)
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid decryption key: {str(e)}"
            )
        
        # Decode base64 data
        try:
            ciphertext_bytes = base64.b64decode(ciphertext)
            iv_bytes = base64.b64decode(iv)
            hmac_bytes = base64.b64decode(hmac)
            metadata_dict = json.loads(metadata)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid input encoding: {str(e)}"
            )
        
        # Create EncryptedPayload
        encrypted_payload = EncryptedPayload(
            ciphertext=ciphertext_bytes,
            iv=iv_bytes,
            hmac=hmac_bytes,
            metadata=metadata_dict,
        )
        
        # Execute decryption pipeline
        reconstructed_points = orchestrator.decrypt_image(
            payload=encrypted_payload,
            key=key,
            visualize=False,
        )
        
        # Convert complex points to list of [x, y] coordinates
        points_list = [
            [float(point.real), float(point.imag)]
            for point in reconstructed_points
        ]
        
        response_data = {
            "status": "success",
            "message": "Image decrypted successfully",
            "reconstructed_points": points_list,
            "point_count": len(points_list),
            "dimensions": metadata_dict.get("dimensions", [0, 0]),
        }
        
        logger.info(
            f"Decryption successful for user: {username}",
            extra={"point_count": len(points_list), "user": username}
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_data
        )
    
    except DecryptionError as e:
        logger.error(f"Decryption error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Decryption failed: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error during decryption: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/visualize")
async def visualize_animation(
    ciphertext: str = Form(..., description="Base64-encoded ciphertext"),
    iv: str = Form(..., description="Base64-encoded initialization vector"),
    hmac: str = Form(..., description="Base64-encoded HMAC"),
    metadata: str = Form(..., description="JSON-encoded metadata"),
    key: str = Form(..., description="Decryption password/key"),
    num_frames: int = Form(100, description="Number of animation frames"),
    username: str = Depends(authenticate),
    orchestrator: EncryptionOrchestrator = Depends(get_orchestrator),
):
    """
    Generate epicycle animation data for visualization.
    
    This endpoint decrypts the payload and generates animation frame data
    showing the epicycle-based sketch reconstruction process.
    
    Args:
        ciphertext: Base64-encoded encrypted data
        iv: Base64-encoded initialization vector
        hmac: Base64-encoded HMAC authentication tag
        metadata: JSON-encoded metadata dictionary
        key: Decryption password
        num_frames: Number of animation frames to generate
        username: Authenticated username (from dependency)
        orchestrator: Encryption orchestrator instance (from dependency)
        
    Returns:
        JSON response containing animation frame data
        
    Raises:
        HTTPException: If visualization generation fails
    """
    logger.info(
        f"Visualization request received from user: {username}",
        extra={"num_frames": num_frames, "user": username}
    )
    
    try:
        import json
        
        # Validate decryption key
        try:
            InputValidator.validate_key(key, min_length=8)
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid decryption key: {str(e)}"
            )
        
        # Validate num_frames
        if not (10 <= num_frames <= 1000):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="num_frames must be between 10 and 1000"
            )
        
        # Decode base64 data
        try:
            ciphertext_bytes = base64.b64decode(ciphertext)
            iv_bytes = base64.b64decode(iv)
            hmac_bytes = base64.b64decode(hmac)
            metadata_dict = json.loads(metadata)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid input encoding: {str(e)}"
            )
        
        # Create EncryptedPayload
        encrypted_payload = EncryptedPayload(
            ciphertext=ciphertext_bytes,
            iv=iv_bytes,
            hmac=hmac_bytes,
            metadata=metadata_dict,
        )
        
        # Decrypt and get coefficients
        # First decrypt to get serialized data
        salt = bytes.fromhex(metadata_dict["salt"])
        derived_key = orchestrator.encryptor.derive_key(key, salt)
        decrypted_data = orchestrator.encryptor.decrypt(encrypted_payload, derived_key)
        
        # Deserialize coefficients
        coefficients, _ = orchestrator.serializer.deserialize(decrypted_data)
        
        # Create epicycle engine
        engine = EpicycleEngine(coefficients)
        
        # Generate animation frames
        frames = list(engine.generate_animation_frames(num_frames))
        
        # Convert frames to JSON-serializable format
        animation_data = []
        for frame in frames:
            frame_data = {
                "time": float(frame.time),
                "positions": [
                    [float(pos.real), float(pos.imag)]
                    for pos in frame.positions
                ],
                "trace_point": [float(frame.trace_point.real), float(frame.trace_point.imag)],
            }
            animation_data.append(frame_data)
        
        response_data = {
            "status": "success",
            "message": "Animation data generated successfully",
            "animation": {
                "frames": animation_data,
                "frame_count": len(animation_data),
                "coefficient_count": len(coefficients),
            },
            "dimensions": metadata_dict.get("dimensions", [0, 0]),
        }
        
        logger.info(
            f"Visualization generated for user: {username}",
            extra={"frame_count": len(animation_data), "user": username}
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_data
        )
    
    except DecryptionError as e:
        logger.error(f"Decryption error during visualization: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Decryption failed: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error during visualization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post(
    "/reconstruct",
    summary="Reconstruct image from encrypted payload",
    description="Decrypt payload and perform image reconstruction with optional animation",
    tags=["reconstruction"],
)
async def reconstruct_image(
    ciphertext: str = Form(..., description="Base64-encoded ciphertext"),
    iv: str = Form(..., description="Base64-encoded initialization vector"),
    hmac: str = Form(..., description="Base64-encoded HMAC"),
    metadata: str = Form(..., description="JSON-encoded metadata"),
    key: str = Form(..., description="Decryption password/key"),
    mode: str = Form("static", description="Reconstruction mode: static or animated"),
    speed: float = Form(1.0, description="Animation speed multiplier (0.1-10.0)"),
    quality: str = Form("balanced", description="Quality mode: fast, balanced, quality"),
    username: str = Depends(authenticate),
    orchestrator: EncryptionOrchestrator = Depends(get_orchestrator),
):
    """
    Reconstruct image from encrypted payload.
    
    This endpoint decrypts the payload and performs image reconstruction
    using the ImageReconstructor module. Supports both static (final image only)
    and animated (epicycle drawing process) modes.
    
    Args:
        ciphertext: Base64-encoded encrypted data
        iv: Base64-encoded initialization vector
        hmac: Base64-encoded HMAC authentication tag
        metadata: JSON-encoded metadata dictionary
        key: Decryption password
        mode: Reconstruction mode (static or animated)
        speed: Animation speed multiplier
        quality: Quality mode (fast, balanced, quality)
        username: Authenticated username (from dependency)
        orchestrator: Encryption orchestrator instance (from dependency)
        
    Returns:
        JSON response containing reconstruction result
        
    Raises:
        HTTPException: If reconstruction fails
    """
    logger.info(
        f"Reconstruction request received from user: {username}",
        extra={"mode": mode, "quality": quality, "user": username}
    )
    
    try:
        import json
        from fourier_encryption.models.data_models import ReconstructionConfig
        
        # Validate decryption key
        try:
            InputValidator.validate_key(key, min_length=8)
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid decryption key: {str(e)}"
            )
        
        # Validate reconstruction parameters
        if mode not in ["static", "animated"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="mode must be 'static' or 'animated'"
            )
        
        if not (0.1 <= speed <= 10.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="speed must be between 0.1 and 10.0"
            )
        
        if quality not in ["fast", "balanced", "quality"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="quality must be 'fast', 'balanced', or 'quality'"
            )
        
        # Decode base64 data
        try:
            ciphertext_bytes = base64.b64decode(ciphertext)
            iv_bytes = base64.b64decode(iv)
            hmac_bytes = base64.b64decode(hmac)
            metadata_dict = json.loads(metadata)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid input encoding: {str(e)}"
            )
        
        # Create EncryptedPayload
        encrypted_payload = EncryptedPayload(
            ciphertext=ciphertext_bytes,
            iv=iv_bytes,
            hmac=hmac_bytes,
            metadata=metadata_dict,
        )
        
        # Decrypt and get coefficients
        salt = bytes.fromhex(metadata_dict["salt"])
        derived_key = orchestrator.encryptor.derive_key(key, salt)
        decrypted_data = orchestrator.encryptor.decrypt(encrypted_payload, derived_key)
        
        # Deserialize coefficients
        coefficients, _ = orchestrator.serializer.deserialize(decrypted_data)
        
        # Create reconstruction config
        recon_config = ReconstructionConfig(
            mode=mode,
            speed=speed,
            quality=quality,
            save_frames=False,
            save_animation=False,
            output_format="mp4",
            output_path=None,
            backend="matplotlib"
        )
        
        # Perform reconstruction
        result = orchestrator.reconstruct_from_coefficients(coefficients, recon_config)
        
        # Convert final image to base64 for response
        import cv2
        _, buffer = cv2.imencode('.png', result.final_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        response_data = {
            "status": "success",
            "message": "Reconstruction completed successfully",
            "reconstruction": {
                "final_image": image_base64,
                "frame_count": result.frame_count,
                "reconstruction_time": result.reconstruction_time,
                "mode": mode,
                "quality": quality,
            },
            "dimensions": metadata_dict.get("dimensions", [0, 0]),
        }
        
        logger.info(
            f"Reconstruction completed for user: {username}",
            extra={
                "frame_count": result.frame_count,
                "reconstruction_time": result.reconstruction_time,
                "user": username
            }
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_data
        )
    
    except DecryptionError as e:
        logger.error(f"Decryption error during reconstruction: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Decryption failed: {str(e)}"
        )
    
    except ConfigurationError as e:
        logger.error(f"Configuration error during reconstruction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reconstruction not configured: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error during reconstruction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# Exception handlers for better error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom handler for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Custom handler for unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "status_code": 500,
        }
    )
