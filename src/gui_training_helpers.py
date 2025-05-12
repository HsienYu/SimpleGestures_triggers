#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Training Helper Functions for GUI

This module contains helper functions for training models in a GUI environment.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import logging
import threading
import queue
import matplotlib
# Configure matplotlib to use a non-interactive backend to avoid conflicts with tkinter
matplotlib.use('Agg')

logger = logging.getLogger('GestureTrigger.GUIHelpers')


class TrainingCallback(tf.keras.callbacks.Callback):
    """Custom TensorFlow callback to update GUI during training."""

    def __init__(self, progress_queue, stop_event):
        super().__init__()
        self.progress_queue = progress_queue
        self.stop_event = stop_event

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch."""
        if self.stop_event.is_set():
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        if logs:
            progress = {
                'epoch': epoch + 1,
                'total_epochs': self.params['epochs'],
                'accuracy': logs.get('accuracy', 0),
                'val_accuracy': logs.get('val_accuracy', 0),
                'loss': logs.get('loss', 0),
                'val_loss': logs.get('val_loss', 0)
            }

            # Calculate progress percentage (0-100)
            progress_percent = (epoch + 1) / self.params['epochs'] * 100
            progress['progress_percent'] = progress_percent

            # Put the progress in the queue for the main thread to process
            self.progress_queue.put(progress)

        # Check if training should be stopped
        if self.stop_event.is_set():
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        """Called at the end of training."""
        # Signal that training is complete
        self.progress_queue.put({'status': 'complete'})
