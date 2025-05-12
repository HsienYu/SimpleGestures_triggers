#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Trainer Module

This module handles training a gesture recognition model using collected data.
"""

import matplotlib.pyplot as plt
import os
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib
# Use Agg backend for matplotlib to avoid conflicts with tkinter
matplotlib.use('Agg')

logger = logging.getLogger('GestureTrigger.ModelTrainer')


class ModelTrainer:
    """Trains a gesture recognition model using collected data."""

    def __init__(self, config):
        """Initialize the model trainer with the given configuration."""
        self.config = config
        self.model_config = config['model']
        self.data_config = config['data_collection']

        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(
            self.model_config['model_path']), exist_ok=True)

    def _load_dataset(self):
        """Load the gesture dataset from disk."""
        dataset_path = self.data_config['dataset_path']

        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return None, None, None

        # Get all gesture classes (folders in dataset path)
        gesture_classes = [d for d in os.listdir(dataset_path)
                           if os.path.isdir(os.path.join(dataset_path, d)) and d != "examples"]

        if not gesture_classes:
            logger.error(f"No gesture classes found in: {dataset_path}")
            return None, None, None

        logger.info(
            f"Found {len(gesture_classes)} gesture classes: {gesture_classes}")

        # Create label mapping
        label_map = {gesture: i for i,
                     gesture in enumerate(sorted(gesture_classes))}

        # Load data and labels
        X = []
        y = []

        for gesture in sorted(gesture_classes):
            gesture_dir = os.path.join(dataset_path, gesture)
            gesture_samples = [f for f in os.listdir(
                gesture_dir) if f.endswith('.npy')]

            logger.info(
                f"Loading {len(gesture_samples)} samples for gesture: {gesture}")

            for sample_file in tqdm(gesture_samples, desc=f"Loading {gesture}"):
                sample_path = os.path.join(gesture_dir, sample_file)
                try:
                    sample_data = np.load(sample_path)

                    # Check if sample has the expected shape/length
                    if len(sample_data.shape) != 1:
                        logger.warning(
                            f"Skipping sample {sample_path} with unexpected dimensions: {sample_data.shape}")
                        continue

                    # Expected feature length: 21*3*2 (hand) + 33*4 (pose) = 258
                    expected_length = 258

                    if sample_data.shape[0] != expected_length:
                        logger.warning(
                            f"Skipping sample {sample_path} with incorrect feature length: {sample_data.shape[0]} (expected {expected_length})")
                        continue

                    X.append(sample_data)
                    y.append(label_map[gesture])
                except Exception as e:
                    logger.warning(f"Failed to load sample {sample_path}: {e}")

        if not X:
            logger.error("No valid samples were loaded")
            return None, None, None

        if not X:
            logger.error("No valid samples were loaded")
            return None, None, None

        # Convert to numpy arrays
        try:
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.int32)

            # Log the shape of the data for debugging
            logger.info(
                f"Dataset loaded successfully. X shape: {X.shape}, y shape: {y.shape}")

            # Handle single class case
            if len(set(y)) == 1:
                logger.info(
                    f"Only one class detected ({list(label_map.keys())[0]}). Converting to binary classification.")
                # For binary classification with only one class, set labels to 0 to fit in valid range
                y = np.zeros_like(y)

            # Save label map to disk
            label_map_file = os.path.join(os.path.dirname(
                self.model_config['model_path']), 'label_map.npy')
            np.save(label_map_file, label_map)

            return X, y, label_map

        except Exception as e:
            logger.error(f"Error converting data to numpy arrays: {e}")
            return None, None, None

    def validate_dataset(self):
        """Validate all samples in the dataset and report any issues."""
        dataset_path = self.data_config['dataset_path']

        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return False

        # Get all gesture classes (folders in dataset path)
        gesture_classes = [d for d in os.listdir(dataset_path)
                           if os.path.isdir(os.path.join(dataset_path, d)) and d != "examples"]

        if not gesture_classes:
            logger.error(f"No gesture classes found in: {dataset_path}")
            return False

        expected_length = 258  # 21*3*2 (hand) + 33*4 (pose) = 258
        issues_found = False
        corrupt_files = []

        logger.info(f"Validating {len(gesture_classes)} gesture classes...")

        for gesture in sorted(gesture_classes):
            gesture_dir = os.path.join(dataset_path, gesture)
            gesture_samples = [f for f in os.listdir(
                gesture_dir) if f.endswith('.npy')]

            logger.info(
                f"Checking {len(gesture_samples)} samples for gesture: {gesture}")

            for sample_file in tqdm(gesture_samples, desc=f"Validating {gesture}"):
                sample_path = os.path.join(gesture_dir, sample_file)
                try:
                    sample_data = np.load(sample_path)

                    # Check if sample has the expected shape/length
                    if len(sample_data.shape) != 1:
                        logger.warning(
                            f"Sample {sample_path} has unexpected dimensions: {sample_data.shape}")
                        corrupt_files.append(sample_path)
                        issues_found = True
                        continue

                    if sample_data.shape[0] != expected_length:
                        logger.warning(
                            f"Sample {sample_path} has incorrect feature length: {sample_data.shape[0]} (expected {expected_length})")
                        corrupt_files.append(sample_path)
                        issues_found = True

                except Exception as e:
                    logger.warning(f"Failed to load sample {sample_path}: {e}")
                    corrupt_files.append(sample_path)
                    issues_found = True

        if issues_found:
            logger.error(
                f"Found {len(corrupt_files)} problematic sample files")
            for file in corrupt_files[:10]:  # Show first 10 to avoid log spam
                logger.error(f"  - {file}")
            if len(corrupt_files) > 10:
                logger.error(f"  ... and {len(corrupt_files) - 10} more")
            return False
        else:
            logger.info("All samples passed validation")
            return True

    def _create_model(self, input_shape, num_classes, learning_rate=0.001):
        """Create a neural network model for gesture recognition."""
        # For multi-class classification (2 or more classes)
        if num_classes > 1:
            model = models.Sequential([
                layers.Input(shape=input_shape),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dense(num_classes, activation='softmax')
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        # For binary classification (only 1 class vs "not that class")
        else:
            model = models.Sequential([
                layers.Input(shape=input_shape),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                # Use sigmoid for binary classification
                layers.Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=learning_rate),
                loss='binary_crossentropy',  # Use binary crossentropy for single class
                metrics=['accuracy']
            )

        return model

    def train_model(self, epochs=100, batch_size=32, learning_rate=0.001, custom_callback=None):
        """Train a gesture recognition model using the collected data."""
        # Load dataset
        logger.info("Loading dataset...")
        X, y, label_map = self._load_dataset()

        if X is None or y is None:
            logger.error("Failed to load dataset")
            return None, None

        logger.info(
            f"Dataset loaded: {X.shape[0]} samples, {len(set(y))} classes")

        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")

        # Create and compile the model
        input_shape = X_train.shape[1:]
        num_classes = len(set(y))

        logger.info(
            f"Creating model with input shape {input_shape} and {num_classes} output classes")
        model = self._create_model(input_shape, num_classes, learning_rate)

        # Print model summary
        model.summary()

        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.model_config['model_path'],
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        # Add custom callback if provided (for GUI progress tracking)
        if custom_callback:
            callbacks.append(custom_callback)

        # Train the model
        logger.info("Training model...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate the model
        logger.info("Evaluating model...")
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Validation loss: {loss:.4f}")
        logger.info(f"Validation accuracy: {accuracy:.4f}")

        # Save the model
        model.save(self.model_config['model_path'])
        logger.info(f"Model saved to: {self.model_config['model_path']}")

        # Save training history plot
        self._plot_training_history(history)

        return model, label_map

    def _plot_training_history(self, history):
        """Plot and save the training history."""
        if not hasattr(history, 'history'):
            return

        # Create plots directory
        plots_dir = os.path.join(os.path.dirname(
            self.model_config['model_path']), 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Plot training & validation accuracy
        plt.figure(figsize=(12, 5))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()

        # Save the plot
        history_plot_path = os.path.join(plots_dir, 'training_history.png')
        plt.savefig(history_plot_path)
        logger.info(f"Training history plot saved to: {history_plot_path}")
        plt.close()
