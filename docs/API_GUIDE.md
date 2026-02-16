# Fourier Image Encryption API

Industrial-grade REST API for image encryption using Fourier Series and epicycles.

## Features

- **Image Encryption**: Encrypt images using Fourier coefficient manipulation
- **Image Decryption**: Decrypt and reconstruct images from encrypted payloads
- **Visualization**: Generate epicycle animation data for visual reconstruction
- **Rate Limiting**: Prevents abuse with configurable request limits
- **Authentication**: HTTP Basic Authentication for secure access
- **OpenAPI Documentation**: Interactive API documentation at `/docs`

## Quick Start

### Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Running the Server

```bash
python run_api.py
```

The server will start on `http://localhost:8000`

### API Documentation

- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Authentication

All endpoints (except `/` and `/health`) require HTTP Basic Authentication.

**Default Credentials** (change in production):
- Username: `admin`
- Password: `secure_password_123`

**Security Note**: In production, use environment variables for credentials:
```python
VALID_USERNAME = os.getenv("API_USERNAME", "admin")
VALID_PASSWORD = os.getenv("API_PASSWORD", "secure_password_123")
```

## Rate Limiting

The API implements rate limiting to prevent abuse:
- **Limit**: 10 requests per 60 seconds per IP address
- **Response**: HTTP 429 (Too Many Requests) when limit exceeded

Configure in `routes.py`:
```python
RATE_LIMIT_REQUESTS = 10  # Maximum requests
RATE_LIMIT_WINDOW = 60    # Time window in seconds
```

## Endpoints

### 1. Root Endpoint

**GET** `/`

Returns API information and available endpoints.

**Response**:
```json
{
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
```

### 2. Health Check

**GET** `/health`

Returns service health status.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": 1234567890.123
}
```

### 3. Encrypt Image

**POST** `/encrypt`

Encrypts an image using Fourier coefficient encryption.

**Authentication**: Required

**Parameters**:
- `image` (file): Image file to encrypt (PNG, JPG, BMP)
- `key` (form): Encryption password/key
- `num_coefficients` (form, optional): Number of Fourier coefficients (10-1000)

**Example using cURL**:
```bash
curl -X POST "http://localhost:8000/encrypt" \
  -u admin:secure_password_123 \
  -F "image=@test_image.png" \
  -F "key=my_secret_key" \
  -F "num_coefficients=100"
```

**Example using Python**:
```python
import requests
from requests.auth import HTTPBasicAuth

url = "http://localhost:8000/encrypt"
auth = HTTPBasicAuth("admin", "secure_password_123")

with open("test_image.png", "rb") as f:
    files = {"image": f}
    data = {
        "key": "my_secret_key",
        "num_coefficients": 100
    }
    response = requests.post(url, auth=auth, files=files, data=data)
    
print(response.json())
```

**Response**:
```json
{
  "status": "success",
  "message": "Image encrypted successfully",
  "encrypted_data": {
    "ciphertext": "base64_encoded_ciphertext...",
    "iv": "base64_encoded_iv...",
    "hmac": "base64_encoded_hmac...",
    "metadata": {
      "salt": "hex_encoded_salt...",
      "kdf_iterations": 100000,
      "dimensions": [1920, 1080],
      "original_contour_length": 500
    }
  },
  "size_bytes": 12345
}
```

### 4. Decrypt Image

**POST** `/decrypt`

Decrypts an encrypted image payload.

**Authentication**: Required

**Parameters**:
- `ciphertext` (form): Base64-encoded ciphertext
- `iv` (form): Base64-encoded initialization vector
- `hmac` (form): Base64-encoded HMAC
- `metadata` (form): JSON-encoded metadata
- `key` (form): Decryption password/key

**Example using Python**:
```python
import requests
import json
from requests.auth import HTTPBasicAuth

url = "http://localhost:8000/decrypt"
auth = HTTPBasicAuth("admin", "secure_password_123")

# Use encrypted_data from encrypt response
data = {
    "ciphertext": encrypted_data["ciphertext"],
    "iv": encrypted_data["iv"],
    "hmac": encrypted_data["hmac"],
    "metadata": json.dumps(encrypted_data["metadata"]),
    "key": "my_secret_key"
}

response = requests.post(url, auth=auth, data=data)
print(response.json())
```

**Response**:
```json
{
  "status": "success",
  "message": "Image decrypted successfully",
  "reconstructed_points": [
    [123.45, 678.90],
    [234.56, 789.01],
    ...
  ],
  "point_count": 500,
  "dimensions": [1920, 1080]
}
```

### 5. Visualize Animation

**POST** `/visualize`

Generates epicycle animation data for visualization.

**Authentication**: Required

**Parameters**:
- `ciphertext` (form): Base64-encoded ciphertext
- `iv` (form): Base64-encoded initialization vector
- `hmac` (form): Base64-encoded HMAC
- `metadata` (form): JSON-encoded metadata
- `key` (form): Decryption password/key
- `num_frames` (form, optional): Number of animation frames (10-1000, default: 100)

**Example using Python**:
```python
import requests
import json
from requests.auth import HTTPBasicAuth

url = "http://localhost:8000/visualize"
auth = HTTPBasicAuth("admin", "secure_password_123")

data = {
    "ciphertext": encrypted_data["ciphertext"],
    "iv": encrypted_data["iv"],
    "hmac": encrypted_data["hmac"],
    "metadata": json.dumps(encrypted_data["metadata"]),
    "key": "my_secret_key",
    "num_frames": 200
}

response = requests.post(url, auth=auth, data=data)
animation_data = response.json()
```

**Response**:
```json
{
  "status": "success",
  "message": "Animation data generated successfully",
  "animation": {
    "frames": [
      {
        "time": 0.0,
        "positions": [[0.0, 0.0], [10.5, 20.3], ...],
        "trace_point": [123.45, 678.90]
      },
      ...
    ],
    "frame_count": 200,
    "coefficient_count": 100
  },
  "dimensions": [1920, 1080]
}
```

## Error Responses

All endpoints return consistent error responses:

**400 Bad Request**:
```json
{
  "error": "Invalid input encoding: ...",
  "status_code": 400
}
```

**401 Unauthorized**:
```json
{
  "error": "Invalid credentials",
  "status_code": 401
}
```

**429 Too Many Requests**:
```json
{
  "error": "Rate limit exceeded",
  "message": "Maximum 10 requests per 60 seconds",
  "retry_after": 60
}
```

**500 Internal Server Error**:
```json
{
  "error": "Internal server error",
  "message": "...",
  "status_code": 500
}
```

## Security Considerations

### Production Deployment

1. **Change Default Credentials**: Use strong, unique credentials
2. **Use HTTPS**: Enable TLS/SSL for encrypted communication
3. **Configure CORS**: Restrict `allow_origins` to trusted domains
4. **Use Redis for Rate Limiting**: Replace in-memory storage with Redis
5. **Environment Variables**: Store sensitive configuration in environment variables
6. **Logging**: Configure structured logging with log rotation
7. **Monitoring**: Set up health checks and performance monitoring

### Example Production Configuration

```python
import os
from redis import Redis

# Authentication from environment
VALID_USERNAME = os.getenv("API_USERNAME")
VALID_PASSWORD = os.getenv("API_PASSWORD")

# Redis for rate limiting
redis_client = Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "").split(","),
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

## Testing

### Manual Testing with cURL

```bash
# Health check
curl http://localhost:8000/health

# Encrypt image
curl -X POST "http://localhost:8000/encrypt" \
  -u admin:secure_password_123 \
  -F "image=@test_image.png" \
  -F "key=test_key_123"

# Test rate limiting (run 11 times quickly)
for i in {1..11}; do
  curl http://localhost:8000/health
done
```

### Automated Testing

See `tests/unit/test_api.py` for comprehensive API tests.

## Performance

- **Encryption**: ~2-5 seconds for 1080p images
- **Decryption**: ~1-3 seconds
- **Visualization**: ~0.5-2 seconds for 100 frames
- **Throughput**: ~10 requests/minute per client (rate limited)

## Troubleshooting

### Server won't start

**Issue**: Port 8000 already in use

**Solution**: Change port in `run_api.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Authentication fails

**Issue**: Invalid credentials

**Solution**: Check username and password in `routes.py` or use environment variables

### Rate limit too restrictive

**Issue**: Getting 429 errors frequently

**Solution**: Adjust rate limit settings in `routes.py`:
```python
RATE_LIMIT_REQUESTS = 20  # Increase limit
RATE_LIMIT_WINDOW = 60
```

## License

See main project LICENSE file.

## Support

For issues and questions, please refer to the main project documentation.
