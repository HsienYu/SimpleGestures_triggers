#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal Gesture Detection Module

This is a simplified version of the gesture detector that works without MediaPipe or TensorFlow.
It provides basic detection capabilities using just OpenCV for systems where the full
dependencies cannot be installed.
"""

import os
import cv2
import time
import logging
import threading
import numpy as np

logger = logging.getLogger('GestureTrigger.BasicDetector')


class BasicGestureDetector:
    """A simplified gesture detector using only OpenCV for basic motion detection."""

    def __init__(self, config):
        """Initialize the basic gesture detector with the given configuration."""
        self.config = config
        self.camera_config = config['camera']

        self.running = False
        self.frame = None
        self.frame_lock = threading.Lock()
        self.detected_gestures = []

        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=50, detectShadows=False)

        # Motion history
        self.motion_history = []
        self.prev_motion = 0

        # Camera thread
        self.camera_thread = None

    def _setup_camera(self):
        """Initialize and configure the camera."""
        device_id = self.camera_config['device_id']
        width = self.camera_config['width']
        height = self.camera_config['height']
        fps = self.camera_config['fps']

        cap = cv2.VideoCapture(device_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        if not cap.isOpened():
            logger.error(f"Failed to open camera (device ID: {device_id})")
            return None

        return cap

    def _detect_motion(self, frame):
        """Detect motion in the frame using background subtraction."""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Calculate the amount of motion
        motion_level = np.sum(fg_mask) / \
            (fg_mask.shape[0] * fg_mask.shape[1] * 255)

        # Add to motion history
        self.motion_history.append(motion_level)
        if len(self.motion_history) > 30:  # Keep 1 second at 30 fps
            self.motion_history.pop(0)

        # Calculate motion change
        motion_change = motion_level - self.prev_motion
        self.prev_motion = motion_level

        # Detect simple gestures based on motion patterns
        gestures = []

        # Detect sudden motion (possible waving)
        if motion_change > 0.05 and motion_level > 0.1:
            gestures.append(("wave", min(motion_level * 5, 0.95)))

        # Detect sustained high motion (possible dancing)
        if len(self.motion_history) > 15 and np.mean(self.motion_history[-15:]) > 0.15:
            gestures.append(("dance", min(motion_level * 4, 0.9)))

        # Detect sudden stop after motion (possible pose)
        if len(self.motion_history) > 10:
            if np.mean(self.motion_history[-10:-5]) > 0.1 and np.mean(self.motion_history[-5:]) < 0.05:
                gestures.append(("pose", 0.8))

        # Return the detected gestures and the foreground mask
        return gestures, fg_mask

    def _camera_loop(self):
        """Background thread for camera capture and processing."""
        logger.info("Starting camera loop")

        # Setup camera
        cap = self._setup_camera()
        if cap is None:
            self.running = False
            return

        try:
            while self.running:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break

                # Mirror the frame for more intuitive feedback
                frame = cv2.flip(frame, 1)

                # Detect motion and gestures
                gestures, motion_mask = self._detect_motion(frame)

                # Update detected gestures
                self.detected_gestures = gestures

                # Create a colored mask for visualization
                motion_display = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

                # Create side-by-side display
                resized_motion = cv2.resize(
                    motion_display, (frame.shape[1] // 3, frame.shape[0] // 3))
                h, w = resized_motion.shape[:2]
                frame[-h:, -w:] = resized_motion

                # Add gesture predictions text
                for i, (gesture, confidence) in enumerate(gestures):
                    threshold = self.config.get('detection', {}).get(
                        'confidence_threshold', 0.7)
                    if confidence > threshold:
                        color = (0, 255, 0)  # Green for high confidence
                    else:
                        color = (0, 165, 255)  # Orange for low confidence

                    cv2.putText(
                        frame,
                        f"{gesture}: {confidence:.2f}",
                        (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

                # Update the shared frame
                with self.frame_lock:
                    self.frame = frame.copy()

                # Display the frame if in debug mode
                if os.environ.get('GESTURE_DEBUG') == '1':
                    cv2.imshow('Basic Gesture Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            cap.release()
            if os.environ.get('GESTURE_DEBUG') == '1':
                cv2.destroyAllWindows()

    def start(self):
        """Start the gesture detector."""
        if self.running:
            logger.warning("Gesture detector is already running")
            return

        self.running = True
        self.camera_thread = threading.Thread(target=self._camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        logger.info("Basic gesture detector started")

    def stop(self):
        """Stop the gesture detector."""
        self.running = False
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
        logger.info("Basic gesture detector stopped")

    def detect(self):
        """Get the current detected gestures."""
        return self.detected_gestures

    def get_frame(self):
        """Get the current frame."""
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
