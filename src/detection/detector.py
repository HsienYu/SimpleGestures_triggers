#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gesture Detector Module

This module handles real-time gesture detection using the trained model.
"""

import os
import cv2
import time
import logging
import threading
import numpy as np
import mediapipe as mp
import tensorflow as tf

logger = logging.getLogger('GestureTrigger.GestureDetector')


class GestureDetector:
    """Detects gestures in real-time using a trained model."""

    def __init__(self, config):
        """Initialize the gesture detector with the given configuration."""
        self.config = config
        self.camera_config = config['camera']
        self.model_config = config['model']
        self.detection_config = config['detection']

        self.running = False
        self.frame = None
        self.frame_lock = threading.Lock()
        self.detected_gestures = []

        # Add detailed logging for debugging
        logger.info("Initializing GestureDetector")
        logger.debug(f"Camera Config: {self.camera_config}")
        logger.debug(f"Model Config: {self.model_config}")
        logger.debug(f"Detection Config: {self.detection_config}")

        # Load label map
        label_map_file = os.path.join(os.path.dirname(
            self.model_config['model_path']), 'label_map.npy')
        if os.path.exists(label_map_file):
            try:
                self.label_map = np.load(
                    label_map_file, allow_pickle=True).item()
                self.reverse_label_map = {
                    v: k for k, v in self.label_map.items()}
                logger.info(
                    f"Loaded label map with {len(self.label_map)} gestures")
                logger.info(
                    f"Label map loaded with gestures: {list(self.label_map.keys())}")
            except Exception as e:
                logger.error(f"Failed to load label map: {e}")
                self.label_map = {}
                self.reverse_label_map = {}
        else:
            logger.warning(f"Label map file not found: {label_map_file}")
            self.label_map = {}
            self.reverse_label_map = {}
            logger.warning(
                "Label map is empty. Ensure the model is trained and label_map.npy exists.")

        # Load model if it exists
        model_path = self.model_config['model_path']
        if os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                logger.info(f"Loaded model from: {model_path}")
                logger.info(f"Model loaded successfully from {model_path}")

                # Get the input shape safely
                try:
                    # Try different ways to get the input shape
                    if hasattr(self.model, 'input_shape'):
                        model_input_size = self.model.input_shape[1]
                    elif hasattr(self.model, 'layers') and len(self.model.layers) > 0:
                        # For first layer, try accessing input shape
                        first_layer = self.model.layers[0]
                        if hasattr(first_layer, 'input_shape'):
                            model_input_size = first_layer.input_shape[1]
                        elif hasattr(first_layer, 'input'):
                            model_input_size = first_layer.input.shape[1]
                        else:
                            # As a fallback, use the configured feature size
                            # Hand landmarks + pose landmarks
                            model_input_size = (21 * 3 * 2) + (33 * 4)
                            logger.warning(
                                f"Could not determine model input shape, using default: {model_input_size}")
                    else:
                        # As a fallback, use the configured feature size
                        # Hand landmarks + pose landmarks
                        model_input_size = (21 * 3 * 2) + (33 * 4)
                        logger.warning(
                            f"Could not determine model input shape, using default: {model_input_size}")

                    self.model_input_size = model_input_size
                except Exception as e:
                    logger.error(f"Error determining model input shape: {e}")
                    # Default to expected size based on feature extraction
                    # Hand landmarks + pose landmarks
                    self.model_input_size = (21 * 3 * 2) + (33 * 4)

                # Check model input shape against expected feature size
                # Hand landmarks + pose landmarks
                expected_feature_size = (21 * 3 * 2) + (33 * 4)
                if self.model_input_size != expected_feature_size:
                    logger.warning(
                        f"Model input shape ({self.model_input_size}) doesn't match expected feature size ({expected_feature_size})")
                    logger.warning(
                        "Feature adaptation will be applied during detection")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None
        else:
            logger.warning(f"Model file not found: {model_path}")
            self.model = None
            logger.warning("No model loaded. Gesture detection will not work.")

        # Initialize MediaPipe solutions
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose

        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5
            )
            logger.debug("MediaPipe hands initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe hands: {e}")
            self.hands = None

        try:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.5
            )
            logger.debug("MediaPipe pose initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe pose: {e}")
            self.pose = None

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

    def _extract_features(self, frame):
        """Extract pose and hand features from a frame."""
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        hand_results = self.hands.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)

        # Extract hand landmarks
        hand_landmarks = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks_obj in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks_obj.landmark:
                    hand_landmarks.extend([landmark.x, landmark.y, landmark.z])

        # If no hands detected, pad with zeros
        while len(hand_landmarks) < 21 * 3 * 2:  # 21 landmarks, 3 coords (x,y,z), 2 hands max
            hand_landmarks.append(0.0)

        # Extract pose landmarks
        pose_landmarks = []
        if pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                pose_landmarks.extend(
                    [landmark.x, landmark.y, landmark.z, landmark.visibility])

        # If no pose detected, pad with zeros
        while len(pose_landmarks) < 33 * 4:  # 33 landmarks, 4 values (x,y,z,visibility)
            pose_landmarks.append(0.0)

        # Combine features
        features = np.array(hand_landmarks + pose_landmarks, dtype=np.float32)

        # Check if model input shape is available and resize features if needed
        if hasattr(self, 'model') and self.model is not None and hasattr(self, 'model_input_size'):
            expected_size = self.model_input_size
            current_size = features.size

            # If this is first detection, log the details only once
            if not hasattr(self, '_feature_size_warned'):
                self._feature_size_warned = True
                if current_size != expected_size:
                    logger.warning(
                        f"Feature size mismatch: got {current_size}, model expects {expected_size}")
                    logger.info(
                        f"This is expected with newer MediaPipe versions. Features will be automatically adjusted.")

            # Resize to match the model's expected input size
            if current_size != expected_size:
                if current_size > expected_size:
                    # Only log this once every 100 frames to avoid spamming logs
                    if not hasattr(self, '_log_counter'):
                        self._log_counter = 0
                    self._log_counter = (self._log_counter + 1) % 100
                    if self._log_counter == 0:
                        logger.info(
                            f"Truncating features from {current_size} to {expected_size}")
                    features = features[:expected_size]
                else:
                    if not hasattr(self, '_log_counter'):
                        self._log_counter = 0
                    self._log_counter = (self._log_counter + 1) % 100
                    if self._log_counter == 0:
                        logger.info(
                            f"Padding features from {current_size} to {expected_size}")
                    padding = np.zeros(
                        expected_size - current_size, dtype=np.float32)
                    features = np.concatenate([features, padding])

        return features, hand_results, pose_results

    def _draw_landmarks(self, frame, hand_results, pose_results):
        """Draw landmarks on the frame for visualization."""
        mp_drawing = mp.solutions.drawing_utils

        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Draw pose landmarks
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return frame

    def _predict_gesture(self, features):
        """Predict the gesture using the loaded model."""
        if self.model is None:
            return []

        # Reshape features for model input
        features_batch = np.expand_dims(features, axis=0)

        # Get model predictions
        predictions = self.model.predict(features_batch, verbose=0)

        # Handle binary classification (single output)
        if len(predictions.shape) == 2 and predictions.shape[1] == 1:
            # For binary classification model with sigmoid activation
            confidence = float(predictions[0][0])
            gesture_name = list(self.reverse_label_map.values())[
                0]  # Use the single class name

            # Apply confidence threshold
            if confidence >= self.detection_config['confidence_threshold']:
                return [(gesture_name, confidence)]
            else:
                return []
        else:
            # Handle multi-class classification with softmax activation
            predictions = predictions[0]

            # Get top predictions
            top_indices = np.argsort(predictions)[::-1]

            results = []
            for idx in top_indices:
                if idx in self.reverse_label_map:
                    gesture_name = self.reverse_label_map[idx]
                    confidence = float(predictions[idx])

                    # Apply confidence threshold
                    if confidence >= self.detection_config['confidence_threshold']:
                        results.append((gesture_name, confidence))

            return results

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

                # Extract features and perform detection
                features, hand_results, pose_results = self._extract_features(
                    frame)

                # Predict gesture
                gesture_predictions = self._predict_gesture(features)

                # Update detected gestures
                self.detected_gestures = gesture_predictions

                # Draw landmarks and gesture info
                annotated_frame = self._draw_landmarks(
                    frame.copy(), hand_results, pose_results)

                # Add gesture predictions text
                # Show top 3
                for i, (gesture, confidence) in enumerate(gesture_predictions[:3]):
                    if confidence > self.detection_config['confidence_threshold']:
                        color = (0, 255, 0)  # Green for high confidence
                    else:
                        color = (0, 165, 255)  # Orange for low confidence

                    cv2.putText(
                        annotated_frame,
                        f"{gesture}: {confidence:.2f}",
                        (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

                # Update the shared frame
                with self.frame_lock:
                    self.frame = annotated_frame.copy()

                # Display the frame if in debug mode
                if os.environ.get('GESTURE_DEBUG') == '1':
                    try:
                        if annotated_frame is None or not isinstance(annotated_frame, (np.ndarray,)):
                            logger.error(
                                "Invalid frame data. Skipping display.")
                            continue

                        cv2.imshow('Gesture Detection', annotated_frame)
                    except cv2.error as e:
                        logger.error(f"OpenCV error during imshow: {e}")
                        continue

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

        # Check if MediaPipe components are initialized
        if not hasattr(self, 'hands') or self.hands is None:
            logger.error(
                "MediaPipe hands not initialized. Cannot start detection.")
            try:
                # Try to reinitialize hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5
                )
                logger.info("Successfully reinitialized MediaPipe hands")
            except Exception as e:
                logger.error(f"Failed to reinitialize MediaPipe hands: {e}")
                return

        if not hasattr(self, 'pose') or self.pose is None:
            logger.error(
                "MediaPipe pose not initialized. Cannot start detection.")
            try:
                # Try to reinitialize pose
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    min_detection_confidence=0.5
                )
                logger.info("Successfully reinitialized MediaPipe pose")
            except Exception as e:
                logger.error(f"Failed to reinitialize MediaPipe pose: {e}")
                return

        # If model is not loaded, gesture detection won't work
        if self.model is None:
            logger.warning("No model loaded. Gesture detection will not work.")

        # Start detector
        self.running = True
        self.camera_thread = threading.Thread(target=self._camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        logger.info("Gesture detector started")

    def stop(self):
        """Stop the gesture detector."""
        self.running = False
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)

        # Safely close MediaPipe objects if they exist
        try:
            if hasattr(self, 'hands') and self.hands:
                self.hands.close()
                logger.debug("MediaPipe hands closed successfully")
        except Exception as e:
            logger.warning(f"Error closing MediaPipe hands: {e}")

        try:
            if hasattr(self, 'pose') and self.pose:
                self.pose.close()
                logger.debug("MediaPipe pose closed successfully")
        except Exception as e:
            logger.warning(f"Error closing MediaPipe pose: {e}")

        logger.info("Gesture detector stopped")

    def detect(self):
        """Get the current detected gestures."""
        return self.detected_gestures

    def get_frame(self):
        """Get the current annotated frame."""
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
