# CLI Usage Guide

## Installation

Install the package with CLI support:

```bash
pip install -e .
```

This will install the `fourier-encrypt` command globally.

## Commands

### Encrypt

Encrypt an image using Fourier-based encryption:

```bash
fourier-encrypt encrypt -i <input_image> -o <output_file> -k <password>
```

**Options:**
- `-i, --input`: Path to input image file (PNG, JPG, BMP) [required]
- `-o, --output`: Path to output encrypted file (JSON format) [required]
- `-k, --key`: Encryption password/key (minimum 8 characters) [required]
- `-c, --config`: Path to configuration file (YAML or JSON) [optional]
- `-n, --coefficients`: Number of Fourier coefficients (10-1000, default: auto) [optional]

**Example:**
```bash
fourier-encrypt encrypt -i images/luffy2.png -o encrypted.json -k MySecurePassword123!
```

**With custom coefficient count:**
```bash
fourier-encrypt encrypt -i images/luffy2.png -o encrypted.json -k MySecurePassword123! -n 100
```

### Decrypt

Decrypt an encrypted image and optionally visualize the reconstruction:

```bash
fourier-encrypt decrypt -i <encrypted_file> -o <output_file> -k <password>
```

**Options:**
- `-i, --input`: Path to encrypted file (JSON format) [required]
- `-o, --output`: Path to output reconstructed contour file (NPY format) [required]
- `-k, --key`: Decryption password/key [required]
- `-v, --visualize`: Enable live epicycle animation during decryption [optional]
- `-c, --config`: Path to configuration file (YAML or JSON) [optional]
- `-s, --speed`: Animation speed multiplier (0.1-10.0, only with --visualize) [optional, default: 4.0]

**Example:**
```bash
fourier-encrypt decrypt -i encrypted.json -o contour.npy -k MySecurePassword123!
```

**With visualization:**
```bash
fourier-encrypt decrypt -i encrypted.json -o contour.npy -k MySecurePassword123! --visualize
```

**With custom animation speed:**
```bash
fourier-encrypt decrypt -i encrypted.json -o contour.npy -k MySecurePassword123! --visualize --speed 6.0
```

### Version

Display version information:

```bash
fourier-encrypt version
```

## Features

### Progress Tracking

The CLI provides real-time progress tracking for long operations:
- Spinner animations during initialization
- Progress bars with time estimates during encryption/decryption
- Detailed status messages for each pipeline stage

### Rich Output

The CLI uses the `rich` library for beautiful terminal output:
- Colored text and panels
- Tables for settings display
- Clear success/error messages
- Formatted help documentation

### Error Handling

The CLI provides clear error messages for common issues:
- Invalid file paths
- Wrong encryption keys
- Configuration errors
- Image processing failures

### Security

- Passwords are prompted securely (hidden input)
- Keys are never logged or displayed
- HMAC verification prevents tampering
- Strong key derivation (PBKDF2 with 100,000+ iterations)

## Configuration File

You can provide a custom configuration file in YAML or JSON format:

**Example config.yaml:**
```yaml
encryption:
  num_coefficients: 100
  use_ai_edge_detection: false
  use_ai_optimization: false
  use_anomaly_detection: false
  kdf_iterations: 100000
  visualization_enabled: false
  animation_speed: 1.0

preprocessing:
  target_size: [1920, 1080]
  maintain_aspect_ratio: true
  normalize: true
  denoise: false
  denoise_strength: 0.5
```

**Usage:**
```bash
fourier-encrypt encrypt -i image.png -o encrypted.json -k password --config config.yaml
```

## Exit Codes

- `0`: Success
- `1`: Error (encryption/decryption failed, invalid configuration, etc.)

## Examples

### Basic Workflow

1. Encrypt an image:
```bash
fourier-encrypt encrypt -i images/luffy2.png -o encrypted.json -k MyPassword123!
```

2. Decrypt without visualization:
```bash
fourier-encrypt decrypt -i encrypted.json -o contour.npy -k MyPassword123!
```

3. Decrypt with visualization:
```bash
fourier-encrypt decrypt -i encrypted.json -o contour.npy -k MyPassword123! --visualize
```

### Advanced Usage

1. Encrypt with custom coefficient count:
```bash
fourier-encrypt encrypt -i image.png -o encrypted.json -k password -n 200
```

2. Decrypt with fast animation:
```bash
fourier-encrypt decrypt -i encrypted.json -o contour.npy -k password --visualize --speed 8.0
```

3. Use custom configuration:
```bash
fourier-encrypt encrypt -i image.png -o encrypted.json -k password --config my_config.yaml
```

## Troubleshooting

### "Configuration error"
- Check that your configuration file is valid YAML or JSON
- Ensure all required fields are present
- Verify file paths exist

### "Encryption key does not meet minimum strength requirements"
- Use a password with at least 8 characters
- Include a mix of letters, numbers, and special characters

### "HMAC verification failed"
- You're using the wrong decryption key
- The encrypted file may be corrupted or tampered with

### "No valid contours found in image"
- The image may be too simple (uniform color)
- Try a different image with more detail
- Check that the image file is not corrupted

## Development

To run the CLI in development mode:

```bash
python -m fourier_encryption.cli.commands --help
```

This allows you to test changes without reinstalling the package.
