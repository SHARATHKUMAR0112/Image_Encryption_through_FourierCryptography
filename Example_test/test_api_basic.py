"""
Basic test script to verify the API implementation.

This script tests that the API can be imported and basic endpoints are defined.
"""

import sys


def test_api_import():
    """Test that the API module can be imported."""
    try:
        from fourier_encryption.api import app
        print("✓ API module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import API module: {e}")
        return False


def test_api_routes():
    """Test that all required routes are defined."""
    try:
        from fourier_encryption.api.routes import app
        
        # Get all routes
        routes = [route.path for route in app.routes]
        
        # Check required endpoints
        required_endpoints = ["/", "/health", "/encrypt", "/decrypt", "/visualize"]
        
        all_present = True
        for endpoint in required_endpoints:
            if endpoint in routes:
                print(f"✓ Endpoint {endpoint} is defined")
            else:
                print(f"✗ Endpoint {endpoint} is missing")
                all_present = False
        
        return all_present
    except Exception as e:
        print(f"✗ Failed to check routes: {e}")
        return False


def test_api_metadata():
    """Test that API metadata is correct."""
    try:
        from fourier_encryption.api.routes import app
        
        # Check app metadata
        assert app.title == "Fourier Image Encryption API", "Incorrect API title"
        assert app.version == "1.0.0", "Incorrect API version"
        assert app.docs_url == "/docs", "Incorrect docs URL"
        assert app.redoc_url == "/redoc", "Incorrect redoc URL"
        
        print("✓ API metadata is correct")
        return True
    except AssertionError as e:
        print(f"✗ API metadata check failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Failed to check API metadata: {e}")
        return False


def test_middleware():
    """Test that middleware is configured."""
    try:
        from fourier_encryption.api.routes import app
        
        # Check that middleware is present
        middleware_count = len(app.user_middleware)
        
        if middleware_count > 0:
            print(f"✓ Middleware configured ({middleware_count} middleware)")
            return True
        else:
            print("✗ No middleware configured")
            return False
    except Exception as e:
        print(f"✗ Failed to check middleware: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Fourier Image Encryption API Implementation")
    print("=" * 60)
    print()
    
    tests = [
        ("Import Test", test_api_import),
        ("Routes Test", test_api_routes),
        ("Metadata Test", test_api_metadata),
        ("Middleware Test", test_middleware),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        result = test_func()
        results.append(result)
    
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
