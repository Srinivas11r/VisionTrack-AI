"""
Installation Verification Script
Run this after installing dependencies to verify everything is set up correctly
"""

import os
import sys


def check_python_version():
    """Check Python version"""
    print("Checking Python version...", end=" ")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} (requires 3.10+)")
        return False


def check_dependencies():
    """Check if all required packages are installed"""
    print("\nChecking dependencies...")

    required_packages = {
        "cv2": "opencv-python",
        "ultralytics": "ultralytics",
        "numpy": "numpy",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "streamlit": "streamlit",
        "sqlalchemy": "sqlalchemy",
        "pandas": "pandas",
        "PIL": "Pillow",
    }

    all_installed = True

    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - Not installed")
            all_installed = False

    return all_installed


def check_model_download():
    """Check if YOLOv8 model can be loaded"""
    print("\nChecking YOLOv8 model...", end=" ")
    try:
        from ultralytics import YOLO

        _ = YOLO("yolov8n.pt")  # Will download if not present
        print("✓ YOLOv8n model ready")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def check_directories():
    """Check if required directories exist"""
    print("\nChecking directories...")

    required_dirs = [
        "backend",
        "backend/core",
        "backend/api",
        "backend/database",
        "backend/utils",
        "frontend",
        "outputs",
        "uploads",
    ]

    all_exist = True

    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ - Missing")
            all_exist = False

    return all_exist


def check_files():
    """Check if core files exist"""
    print("\nChecking core files...")

    required_files = [
        "backend/core/detector.py",
        "backend/core/tracker.py",
        "backend/core/processor.py",
        "backend/api/main.py",
        "backend/utils/filters.py",
        "backend/database/models.py",
        "frontend/app.py",
        "requirements.txt",
    ]

    all_exist = True

    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - Missing")
            all_exist = False

    return all_exist


def test_imports():
    """Test if backend modules can be imported"""
    print("\nTesting module imports...")

    tests = [
        ("backend.core.detector", "ObjectDetector"),
        ("backend.core.tracker", "ObjectTracker"),
        ("backend.core.processor", "VideoProcessor"),
        ("backend.utils.filters", "detect_color"),
        ("backend.database.models", "Database"),
    ]

    all_passed = True

    for module_name, class_name in tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"  ✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"  ✗ {module_name}.{class_name} - {e}")
            all_passed = False

    return all_passed


def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")

    try:
        # Test detector
        print("  Testing detector...", end=" ")
        from backend.core.detector import ObjectDetector

        _ = ObjectDetector()
        print("✓")

        # Test tracker
        print("  Testing tracker...", end=" ")
        from backend.core.tracker import ObjectTracker

        _ = ObjectTracker()
        print("✓")

        # Test processor
        print("  Testing processor...", end=" ")
        from backend.core.processor import VideoProcessor

        _ = VideoProcessor()
        print("✓")

        # Test filters
        print("  Testing color detection...", end=" ")
        import numpy as np

        from backend.utils.filters import classify_size, detect_color

        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        detect_color(test_img, [0, 0, 100, 100])
        classify_size([0, 0, 100, 100])
        print("✓")

        return True

    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        return False


def check_gpu():
    """Check if GPU is available"""
    print("\nChecking GPU availability...")
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✓ GPU available: {gpu_name}")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
        else:
            print("  ℹ GPU not available (will use CPU)")
    except ImportError:
        print("  ℹ PyTorch not installed (GPU support unavailable)")


def check_webcam():
    """Check if webcam is accessible"""
    print("\nChecking webcam...")
    try:
        import cv2

        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("  ✓ Webcam accessible")
            cap.release()
        else:
            print("  ℹ Webcam not accessible (optional)")
    except Exception as e:
        print(f"  ℹ Webcam check failed: {e}")


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\n1. Start the backend server:")
    print("   cd backend")
    print("   python -m uvicorn api.main:app --reload --port 8000")
    print("\n2. In a new terminal, start the frontend:")
    print("   streamlit run frontend/app.py")
    print("\n3. Open your browser to: http://localhost:8501")
    print("\n4. Or test standalone:")
    print("   python test_processor.py")
    print("   python examples.py")
    print("\n5. Read the documentation:")
    print("   - QUICKSTART.md for usage guide")
    print("   - API_DOCS.md for API reference")
    print("   - DEVELOPMENT.md for development tips")
    print("\n" + "=" * 60)


def main():
    """Run all checks"""
    print("=" * 60)
    print("INSTALLATION VERIFICATION")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directories", check_directories),
        ("Core Files", check_files),
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("YOLOv8 Model", check_model_download),
    ]

    results = []

    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check failed: {e}")
            results.append((name, False))

    # Optional checks
    check_gpu()
    check_webcam()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✓ ALL CHECKS PASSED!")
        print("System is ready to use.")
        print_next_steps()
        return 0
    else:
        print("\n✗ SOME CHECKS FAILED")
        print("Please fix the issues above before running the system.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Check Python version: python --version")
        print("  - Verify file structure matches documentation")
        return 1


if __name__ == "__main__":
    exit(main())
