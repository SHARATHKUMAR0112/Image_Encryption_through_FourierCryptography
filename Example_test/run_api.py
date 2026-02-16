"""
Script to run the Fourier Image Encryption API server.

Usage:
    python run_api.py

The server will start on http://localhost:8000
API documentation available at http://localhost:8000/docs
"""

import logging
import sys

import uvicorn

from fourier_encryption.api import app


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Starting Fourier Image Encryption API server...")
    logger.info("API documentation: http://localhost:8000/docs")
    logger.info("ReDoc documentation: http://localhost:8000/redoc")
    logger.info("Health check: http://localhost:8000/health")
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
    )
