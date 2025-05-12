#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup Script for Gesture Trigger

This script helps set up the development environment for the Gesture Trigger application.
It creates necessary directories, installs dependencies, and checks system requirements.
"""

import os
import sys
import subprocess
import shutil
import platform
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GestureTrigger.Setup')


def check_python_version():
    """Check if the Python version is compatible."""
    required_version = (3, 8)
    current_version = sys.version_info

    if current_version < required_version:
        logger.error(
            f"Python {required_version[0]}.{required_version[1]} or higher is required")
        logger.error(
            f"Current version: {current_version[0]}.{current_version[1]}")
        return False

    logger.info(
        f"Python version check passed: {current_version[0]}.{current_version[1]}")
    return True


def check_camera():
    """Check if a camera is available."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.warning("No camera detected")
            return False

        ret, frame = cap.read()
        cap.release()

        if not ret:
            logger.warning("Camera test failed")
            return False

        logger.info("Camera check passed")
        return True

    except ImportError:
        logger.warning("OpenCV not installed, skipping camera check")
        return True  # Skip check if OpenCV is not installed yet
    except Exception as e:
        logger.warning(f"Camera check failed: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        'data/gesture_dataset',
        'models',
        'assets/sounds',
        'assets/images',
        'logs'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

    return True


def install_dependencies():
    """Install Python dependencies."""
    try:
        # Check if pip is available and update it
        subprocess.run([sys.executable, "-m", "pip", "install",
                       "--upgrade", "pip"], check=True)

        # Determine Python version compatibility
        current_version = sys.version_info

        # For Python 3.12+ on macOS arm64 (Apple Silicon), use alternative packages
        if current_version >= (3, 12) and platform.system() == "Darwin" and platform.machine() == "arm64":
            logger.info("Detected Python 3.12+ on macOS Apple Silicon")
            logger.info("Installing compatible dependencies...")

            # Install basic requirements first
            subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python", "numpy<2.0.0",
                           "PyYAML", "tqdm", "matplotlib", "scikit-learn", "pillow"], check=True)

            # Try MediaPipe Silicon
            try:
                subprocess.run([sys.executable, "-m", "pip", "install",
                               "mediapipe-silicon>=0.9.2"], check=True)
                logger.info("Successfully installed mediapipe-silicon")
            except subprocess.CalledProcessError:
                logger.warning(
                    "Could not install mediapipe-silicon. Some features may be unavailable.")

            # Try TensorFlow for macOS
            try:
                subprocess.run([sys.executable, "-m", "pip", "install",
                               "tensorflow-macos>=2.12.0"], check=True)
                logger.info("Successfully installed tensorflow-macos")
            except subprocess.CalledProcessError:
                logger.warning(
                    "Could not install tensorflow-macos. Model training may be unavailable.")

            # Install appropriate Qt depending on Python version
            if current_version >= (3, 13):
                try:
                    subprocess.run([sys.executable, "-m", "pip",
                                   "install", "PyQt6>=6.5.0"], check=True)
                except subprocess.CalledProcessError:
                    logger.warning(
                        "Could not install PyQt6. GUI may be unavailable.")
            else:
                try:
                    subprocess.run([sys.executable, "-m", "pip",
                                   "install", "PyQt5>=5.15.0"], check=True)
                except subprocess.CalledProcessError:
                    logger.warning(
                        "Could not install PyQt5. GUI may be unavailable.")

        else:
            # Standard installation from requirements.txt
            subprocess.run([sys.executable, "-m", "pip", "install",
                           "-r", "requirements.txt"], check=True)

        logger.info("Successfully installed dependencies")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def check_gpu():
    """Check if TensorFlow can use GPU acceleration."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            logger.info(f"GPU is available: {len(gpus)} device(s)")
            for gpu in gpus:
                logger.info(f"  - {gpu}")
            return True
        else:
            logger.warning("No GPU detected, the application will run on CPU")
            return False

    except ImportError:
        logger.warning("TensorFlow not installed, skipping GPU check")
        return True  # Skip check if TensorFlow is not installed yet
    except Exception as e:
        logger.warning(f"GPU check failed: {e}")
        return False


def create_sample_sound():
    """Create a sample sound file."""
    try:
        import numpy as np
        from scipy.io import wavfile

        # Create a simple sine wave
        sample_rate = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(
            sample_rate * duration), endpoint=False)

        # Generate a short "ding" sound
        frequency = 440
        amplitude = 32767
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)

        # Add a quick fade-out
        fade_out = np.linspace(1.0, 0.0, int(sample_rate * 0.1))
        sine_wave[-len(fade_out):] *= fade_out

        # Convert to 16-bit PCM
        sine_wave = sine_wave.astype(np.int16)

        # Save the sound file
        sound_path = "assets/sounds/chime.wav"
        wavfile.write(sound_path, sample_rate, sine_wave)

        logger.info(f"Created sample sound file: {sound_path}")
        return True

    except ImportError:
        logger.warning("SciPy not installed, skipping sample sound creation")
        return True
    except Exception as e:
        logger.warning(f"Failed to create sample sound: {e}")
        return False


def main():
    """Main setup function."""
    logger.info("=== Gesture Trigger Setup ===")

    # Check Python version
    if not check_python_version():
        logger.error("Setup failed: Python version check failed")
        return False

    # Create directories
    if not create_directories():
        logger.error("Setup failed: Could not create directories")
        return False

    # Install dependencies
    if not install_dependencies():
        logger.error("Setup failed: Could not install dependencies")
        return False

    # Check camera
    check_camera()  # Just a warning, don't fail

    # Check GPU
    check_gpu()  # Just a warning, don't fail

    # Create sample sound
    create_sample_sound()  # Just a warning, don't fail

    logger.info("=== Setup Completed Successfully ===")
    logger.info("You can now run the application with:")
    logger.info(
        "  python main.py --mode collect --gesture your_gesture  # To collect data")
    logger.info(
        "  python main.py --mode train                          # To train a model")
    logger.info(
        "  python main.py --mode run                            # To run gesture detection")
    logger.info("Or use the GUI:")
    logger.info("  python gui.py")

    return True


if __name__ == "__main__":
    main()
