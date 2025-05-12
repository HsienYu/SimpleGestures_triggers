#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gesture Trigger Main Application

This is the main entry point for the Gesture Trigger application.
It integrates all components (data collection, model, detection, triggers)
and provides a command-line interface.
"""

import os
import sys
import argparse
import yaml
import time
import logging
import cv2  # Add OpenCV import for displaying camera feed

from src.data_collection.collector import DataCollector
from src.model.trainer import ModelTrainer
from src.detection.detector import GestureDetector
from src.triggers.trigger_manager import TriggerManager

# Import basic detector for fallback
try:
    from src.detection.basic_detector import BasicGestureDetector
    BASIC_DETECTOR_AVAILABLE = True
except ImportError:
    BASIC_DETECTOR_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GestureTrigger')


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Gesture Trigger Application')
    parser.add_argument(
        '--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--mode', choices=['collect', 'train', 'run', 'debug'], default='run',
                        help='Application mode: collect data, train model, run detection, or debug visually')
    parser.add_argument('--gesture', help='Gesture name for data collection')
    parser.add_argument(
        '--device', help='Override camera device ID from config')
    parser.add_argument('--use-basic-detection', action='store_true',
                        help='Use basic detection mode without MediaPipe or TensorFlow')
    parser.add_argument('--debug-level', choices=['basic', 'advanced'], default='basic',
                        help='Debug level when using debug mode')
    parser.add_argument('--show-camera', action='store_true',
                        help='Show camera feed in run mode')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments if specified
    if args.device is not None:
        config['camera']['device_id'] = int(args.device)

    # Execute the requested mode
    if args.mode == 'collect':
        if not args.gesture:
            logger.error(
                "Gesture name must be specified for data collection mode")
            sys.exit(1)
        collector = DataCollector(config)
        collector.collect_data(args.gesture)

    elif args.mode == 'train':
        trainer = ModelTrainer(config)
        trainer.train_model()

    elif args.mode == 'run':
        # Initialize components
        try:
            # Use basic detector if specified or if full detector is unavailable
            if args.use_basic_detection or not BASIC_DETECTOR_AVAILABLE:
                if args.use_basic_detection:
                    logger.info("Using basic detection mode as requested")
                else:
                    logger.warning(
                        "Full gesture detector unavailable, falling back to basic detection")

                if not BASIC_DETECTOR_AVAILABLE:
                    logger.error(
                        "Basic detector not available. Check your installation.")
                    sys.exit(1)

                detector = BasicGestureDetector(config)
            else:
                try:
                    # Initialize detector with possibly lowered threshold for testing
                    if 'detection' not in config:
                        config['detection'] = {}

                    # Store original threshold value to restore later
                    original_threshold = config['detection'].get(
                        'confidence_threshold', 0.7)

                    # Temporarily decrease threshold for initialization tests
                    # Use 70% of original threshold for testing
                    test_threshold = original_threshold * 0.7
                    config['detection']['confidence_threshold'] = test_threshold

                    detector = GestureDetector(config)

                    # Restore original threshold after initialization
                    config['detection']['confidence_threshold'] = original_threshold

                    logger.info(
                        f"Using lowered test threshold ({test_threshold:.2f}) for initial detection check")

                    # Verify gesture labels in label_map
                    if hasattr(detector, 'label_map'):
                        logger.info(f"Loaded label map: {detector.label_map}")
                        if 'disagree' not in detector.label_map.values():
                            logger.warning(
                                "Gesture 'disagree' not found in label map. Ensure it was included during training.")
                        if 'shooting' not in detector.label_map.values():
                            logger.warning(
                                "Gesture 'shooting' not found in label map. Ensure it was included during training.")

                    # Temporarily lower confidence threshold for debugging
                    debug_threshold = original_threshold * 0.5  # Lower threshold to 50% of original
                    config['detection']['confidence_threshold'] = debug_threshold
                    logger.info(
                        f"Temporarily lowered confidence threshold to {debug_threshold:.2f} for debugging.")

                    # Restore original threshold after initialization
                    config['detection']['confidence_threshold'] = original_threshold

                    # Test the detector immediately to see if it can detect anything
                    detector.start()
                    logger.info(
                        "Testing detector for gesture detection capabilities...")

                    # Try to detect gestures several times with a relaxed threshold
                    detection_success = False
                    for i in range(5):  # Try 5 times
                        time.sleep(0.5)  # Wait between attempts
                        test_gestures = detector.detect()
                        if test_gestures:
                            logger.info(
                                f"Detector test successful! Detected: {test_gestures}")
                            detection_success = True
                            break

                    if not detection_success:
                        logger.warning(
                            "Detector initialization test did not detect any gestures")
                        logger.info(
                            "This may indicate issues with the model or training data")
                except Exception as e:
                    logger.error(f"Failed to initialize full detector: {e}")
                    if BASIC_DETECTOR_AVAILABLE:
                        logger.info("Falling back to basic detector")
                        detector = BasicGestureDetector(config)
                    else:
                        logger.error(
                            "No detectors available. Check your installation.")
                        sys.exit(1)

            # Print detector information for debugging
            logger.info(f"Using detector: {detector.__class__.__name__}")

            # Check if detector has the necessary gestures loaded
            if hasattr(detector, 'get_gesture_names'):
                gestures = detector.get_gesture_names()
                logger.info(f"Available gestures in detector: {gestures}")
            elif hasattr(detector, 'gesture_names'):
                logger.info(
                    f"Available gestures in detector: {detector.gesture_names}")
            else:
                logger.warning(
                    "Could not determine available gestures from detector")

            # Check if model files exist (if applicable)
            if hasattr(detector, 'model_path'):
                model_path = detector.model_path
                if os.path.exists(model_path):
                    logger.info(f"Model file exists: {model_path}")
                    # Get file size for additional debugging
                    model_size = os.path.getsize(
                        model_path) / (1024 * 1024)  # Size in MB
                    logger.info(f"Model file size: {model_size:.2f} MB")
                else:
                    logger.error(f"Model file not found: {model_path}")
                    logger.error(
                        "Please train your model with: python main.py --mode train")

            # Add detailed info about what gestures are being looked for
            if hasattr(detector, 'get_detector_details'):
                details = detector.get_detector_details()
                logger.info(f"Detector details: {details}")

            trigger_manager = TriggerManager(config)

            # Check trigger manager configuration
            if hasattr(trigger_manager, 'get_configured_triggers'):
                triggers = trigger_manager.get_configured_triggers()
                logger.info(f"Configured triggers: {triggers}")
            elif hasattr(trigger_manager, 'triggers'):
                logger.info(
                    f"Configured triggers: {list(trigger_manager.triggers.keys())}")
            else:
                logger.warning("Could not determine configured triggers")

            # Run the main detection loop
            try:
                detector.start()
                logger.info("Gesture detection started. Press 'q' to quit.")

                # Whether to show camera in run mode - default to True for better debugging
                show_camera = args.show_camera or config.get(
                    'display', {}).get('show_camera_in_run_mode', True)

                # Add flag to show all gestures, even below threshold
                show_all_gestures = config.get(
                    'display', {}).get('show_all_gestures', True)

                # Get threshold from config
                threshold = config['detection']['confidence_threshold']
                logger.info(f"Confidence threshold set to {threshold}")

                # For tracking performance
                frame_count = 0
                start_time = time.time()
                last_gesture_time = None
                detected_gestures_history = []  # Keep history of recent detected gestures

                while True:
                    # Get detected gestures
                    gestures = detector.detect()

                    # Log detected gestures explicitly
                    if gestures:
                        logger.info(f"Detected gestures: {gestures}")
                    else:
                        logger.info("No gestures detected")

                    # Print raw data from detector periodically to diagnose issues
                    if frame_count % 10 == 0:
                        # Try to get raw data if available
                        if hasattr(detector, 'get_raw_data'):
                            raw_data = detector.get_raw_data()
                            logger.debug(f"Raw detector data: {raw_data}")

                        # Check camera/detector state
                        if hasattr(detector, 'is_camera_working') and not detector.is_camera_working():
                            logger.error(
                                "Camera not providing frames. Check camera connection.")

                        # Print empty gesture list explicitly for debugging
                        if not gestures:
                            logger.debug(
                                "Detector returned empty gesture list")

                    frame_count += 1

                    # Log all detected gestures periodically (every 50 frames)
                    if frame_count % 50 == 0 and show_all_gestures:
                        raw_gestures_str = ", ".join(
                            [f"{g[0]}:{g[1]:.2f}" for g in gestures])
                        if raw_gestures_str:
                            logger.info(f"Raw gestures: {raw_gestures_str}")
                        else:
                            logger.info("No gestures detected")

                        # Print FPS every 50 frames
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                        logger.info(f"FPS: {fps:.2f}")

                        # Reset counters
                        frame_count = 0
                        start_time = time.time()

                    # Get frame from detector if available
                    frame = None
                    if show_camera and hasattr(detector, 'get_frame'):
                        frame = detector.get_frame()
                        if frame is None:
                            logger.warning(
                                "get_frame() returned None. Camera may not be working.")
                    elif show_camera:
                        logger.warning(
                            "Detector does not support get_frame() method. Cannot show camera feed.")
                        show_camera = False

                    # Check if any gestures are above threshold
                    above_threshold = False

                    # Trigger functions based on detected gestures
                    for gesture_name, confidence in gestures:
                        if confidence > threshold:
                            above_threshold = True
                            detected_gestures_history.append(
                                (gesture_name, confidence, time.time()))

                            # Only keep recent history (last 5 seconds)
                            detected_gestures_history = [g for g in detected_gestures_history
                                                         if time.time() - g[2] < 5]

                            logger.info(
                                f"Detected gesture: {gesture_name} with confidence: {confidence:.2f}")
                            trigger_manager.execute_trigger(gesture_name)
                            last_gesture_time = time.time()

                    # Log when no gestures detected after some previous detection
                    if not above_threshold and last_gesture_time and time.time() - last_gesture_time > 2.0:
                        logger.info("No gestures detected above threshold")
                        last_gesture_time = None

                    # Display the frame if available
                    if show_camera and frame is not None:
                        # Create a copy of the frame for drawing
                        display_frame = frame.copy()

                        # Add a status bar at the top
                        cv2.rectangle(display_frame, (0, 0),
                                      (display_frame.shape[1], 40), (0, 0, 0), -1)

                        # Show threshold info
                        threshold_text = f"Threshold: {threshold:.2f}"
                        cv2.putText(display_frame, threshold_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        # Draw all gestures on frame
                        y_offset = 70
                        for idx, (gesture_name, confidence) in enumerate(gestures):
                            color = (0, 255, 0) if confidence > threshold else (
                                0, 165, 255)
                            text = f"{gesture_name}: {confidence:.2f}"
                            cv2.putText(display_frame, text, (10, y_offset + idx * 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        # Draw a large indicator for the most confident gesture above threshold
                        if gestures:
                            # Sort by confidence
                            sorted_gestures = sorted(
                                gestures, key=lambda x: x[1], reverse=True)
                            top_gesture, top_confidence = sorted_gestures[0]

                            if top_confidence > threshold:
                                # Draw a big green indicator at the bottom
                                cv2.rectangle(display_frame,
                                              (0, display_frame.shape[0]-60),
                                              (display_frame.shape[1],
                                               display_frame.shape[0]),
                                              (0, 255, 0), -1)
                                cv2.putText(display_frame, f"DETECTED: {top_gesture.upper()}",
                                            (display_frame.shape[1]//2 -
                                             150, display_frame.shape[0]-20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                        # Show historical detections
                        if detected_gestures_history:
                            history_y = y_offset + len(gestures) * 30 + 20
                            cv2.putText(display_frame, "Recent Detections:", (10, history_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                            for idx, (g_name, g_conf, g_time) in enumerate(detected_gestures_history[-5:]):
                                # Calculate how long ago the gesture was detected
                                time_ago = time.time() - g_time
                                history_text = f"{g_name} ({time_ago:.1f}s ago)"
                                cv2.putText(display_frame, history_text, (10, history_y + 30 + idx * 25),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                        # Add more diagnostic info to the display
                        detector_name = detector.__class__.__name__
                        cv2.putText(display_frame, f"Detector: {detector_name}",
                                    (display_frame.shape[1] - 300, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                        # Add a counter for frames with no gestures
                        if not gestures:
                            no_gesture_text = "No gestures detected"
                            cv2.putText(display_frame, no_gesture_text,
                                        (display_frame.shape[1]//2 - 100, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # Display camera resolution
                        resolution_text = f"Resolution: {frame.shape[1]}x{frame.shape[0]}"
                        cv2.putText(display_frame, resolution_text, (10, display_frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        cv2.imshow('Gesture Detection', display_frame)

                        # Exit if 'q' is pressed
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("User quit the application")
                            break

                    # Small delay to avoid maxing out CPU
                    time.sleep(0.01)

            except KeyboardInterrupt:
                logger.info("Application stopped by user")

        except ModuleNotFoundError as e:
            logger.error(f"Required module not found: {e}")
            logger.error(
                "Please run the setup script to install necessary dependencies:")
            logger.error("  ./setup.py")
            logger.error("Or try the fallback installation:")
            logger.error("  ./fallback_setup.py")
            sys.exit(1)
        finally:
            detector.stop()
            trigger_manager.cleanup()
            if args.show_camera:
                cv2.destroyAllWindows()

    elif args.mode == 'debug':
        # Set debug environment variable
        os.environ['GESTURE_DEBUG'] = '1'

        # Run the appropriate debugger based on the debug level
        if args.debug_level == 'advanced':
            try:
                # Import and run the visual debugger
                from visual_debugger import VisualDebugger
                logger.info("Starting advanced visual debugger")
                debugger = VisualDebugger(config)
                debugger.run()
            except ImportError:
                logger.error(
                    "Visual debugger not found. Make sure visual_debugger.py exists.")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Error running visual debugger: {e}")
                logger.info("Falling back to basic debugging mode")
                # Fall back to basic debugging if advanced fails
                args.debug_level = 'basic'

        if args.debug_level == 'basic':
            # Use standard detector with debug flag set
            try:
                if args.use_basic_detection or not BASIC_DETECTOR_AVAILABLE:
                    detector = BasicGestureDetector(config)
                else:
                    detector = GestureDetector(config)

                trigger_manager = TriggerManager(config)

                try:
                    detector.start()
                    logger.info(
                        "Basic debug mode active. Press Ctrl+C to exit.")
                    while True:
                        # Get detected gestures
                        gestures = detector.detect()

                        # Log detected gestures
                        if gestures:
                            logger.info(f"Detected gestures: {gestures}")

                            # Trigger functions based on detected gestures
                            for gesture_name, confidence in gestures:
                                if confidence > config['detection']['confidence_threshold']:
                                    logger.info(
                                        f"Triggering action for: {gesture_name} ({confidence:.2f})")
                                    trigger_manager.execute_trigger(
                                        gesture_name)

                        # Small delay to avoid maxing out CPU
                        time.sleep(0.1)

                except KeyboardInterrupt:
                    logger.info("Debug mode stopped by user")
                finally:
                    detector.stop()
                    trigger_manager.cleanup()

            except Exception as e:
                logger.error(f"Error in basic debug mode: {e}")
                sys.exit(1)


if __name__ == "__main__":
    main()
