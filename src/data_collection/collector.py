#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Collector Module

This module handles capturing and saving gesture data for training.
"""

import os
import cv2
import time
import logging
import mediapipe as mp
import numpy as np
from tqdm import tqdm

logger = logging.getLogger('GestureTrigger.DataCollector')


class DataCollector:
    """Collects gesture data for model training."""

    def __init__(self, config):
        """Initialize the data collector with the given configuration."""
        self.config = config
        self.camera_config = config['camera']
        self.data_config = config['data_collection']

        # Create dataset directory if it doesn't exist
        os.makedirs(self.data_config['dataset_path'], exist_ok=True)

        # Initialize MediaPipe solutions
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5
        )

    def _setup_camera(self):
        """Initialize and configure the camera using settings from config.yaml."""
        device_id = self.camera_config['device_id']
        width = self.camera_config['width']
        height = self.camera_config['height']
        fps = self.camera_config['fps']

        logger.info(
            f"Setting up camera with device ID: {device_id} (from config.yaml)")
        logger.info(f"Camera resolution: {width}x{height}, FPS: {fps}")

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

        # Standardize to expected model size - compatibility with older models
        # Standard size for previous models (21*3*2 + 33*4)
        expected_feature_size = 258
        current_size = features.size

        if current_size != expected_feature_size:
            # Handle size difference
            if current_size > expected_feature_size:
                # Truncate features to match expected size
                features = features[:expected_feature_size]
            else:
                # Pad with zeros if necessary (unlikely case)
                padding = np.zeros(expected_feature_size -
                                   current_size, dtype=np.float32)
                features = np.concatenate([features, padding])

        return features

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

    def collect_data(self, gesture_name):
        """Collect data for the specified gesture using camera from config.yaml."""
        logger.info(f"Collecting data for gesture: {gesture_name}")
        logger.info(
            f"Using camera device ID: {self.camera_config['device_id']} from config.yaml")
        logger.info(
            f"Camera resolution: {self.camera_config['width']}x{self.camera_config['height']}")

        # Setup camera
        cap = self._setup_camera()
        if cap is None:
            logger.error(
                "Failed to set up camera. Please check your camera connection and the device ID in config.yaml")
            logger.info(
                "You can override the camera device ID using the --device flag: python main.py --mode collect --gesture NAME --device ID")
            return

        # Create gesture directory
        gesture_dir = os.path.join(
            self.data_config['dataset_path'], gesture_name)
        os.makedirs(gesture_dir, exist_ok=True)

        # Count existing samples to avoid overwriting
        existing_samples = len([f for f in os.listdir(gesture_dir)
                               if f.endswith('.npy')])

        frames_to_collect = self.data_config['frames_per_gesture']
        collected_frames = 0

        logger.info(
            f"Press 'Space' to start recording {frames_to_collect} frames")
        logger.info(f"Press 'Q' to quit")

        recording = False
        pbar = None

        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break

                # Mirror the frame for more intuitive feedback
                frame = cv2.flip(frame, 1)

                if recording:
                    # Extract features
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    hand_results = self.hands.process(rgb_frame)
                    pose_results = self.pose.process(rgb_frame)

                    # Draw landmarks
                    annotated_frame = self._draw_landmarks(
                        frame.copy(), hand_results, pose_results)

                    # Extract and save features
                    features = self._extract_features(frame)
                    sample_path = os.path.join(
                        gesture_dir, f"{gesture_name}_{existing_samples + collected_frames}.npy")
                    np.save(sample_path, features)

                    collected_frames += 1
                    pbar.update(1)

                    # Display recording indicator
                    cv2.putText(annotated_frame, f"Recording: {collected_frames}/{frames_to_collect}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Check if we've collected enough frames
                    if collected_frames >= frames_to_collect:
                        recording = False
                        logger.info(
                            f"Finished collecting data for gesture: {gesture_name}")
                        pbar.close()
                        pbar = None
                else:
                    # Just show the camera feed with a prompt
                    annotated_frame = frame.copy()
                    cv2.putText(annotated_frame, "Press SPACE to start recording",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display the frame
                cv2.imshow('Data Collection', annotated_frame)

                # Check for user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' ') and not recording:
                    recording = True
                    collected_frames = 0
                    logger.info(f"Started recording gesture: {gesture_name}")
                    pbar = tqdm(total=frames_to_collect,
                                desc=f"Collecting {gesture_name}")

        finally:
            if pbar:
                pbar.close()
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            self.pose.close()

            logger.info(
                f"Collected {collected_frames} frames for gesture: {gesture_name}")
            logger.info(
                f"Total samples for {gesture_name}: {existing_samples + collected_frames}")
