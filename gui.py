#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gesture Trigger GUI

A graphical user interface for the Gesture Trigger application.
"""

from src.triggers.trigger_manager import TriggerManager
from src.detection.detector import GestureDetector
from src.model.trainer import ModelTrainer
from src.data_collection.collector import DataCollector
from src.gui_helpers import add_write_gesture_export_info, process_import_dataset
from src.gui_trigger_helpers import TriggerLogManager
import os
import sys
import time
import threading
import queue  # Add queue import for thread communication
import yaml
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import json
import shutil
from datetime import datetime

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GestureTrigger.GUI')


class GestureTriggerGUI:
    """GUI for the Gesture Trigger application."""

    def __init__(self, config_path='config/config.yaml'):
        """Initialize the GUI with the given configuration."""
        self.root = tk.Tk()
        self.root.title("Gesture Trigger")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        # Set custom theme
        self._set_custom_style()

        # Load configuration
        self.config_path = config_path
        self.config = self._load_config()

        # Application state
        self.running = False
        self.recording = False
        self.countdown_active = False
        self.countdown_seconds = 5
        self.countdown_time = 0
        self.collected_frames = 0
        self.frames_to_collect = 0
        self.current_mode = tk.StringVar(value="run")
        self.current_gesture = tk.StringVar(value="")
        self.detector = None
        self.trigger_manager = None
        self.video_thread = None
        self.stop_event = threading.Event()

        # Available gestures
        self.available_gestures = self._load_available_gestures()

        # Training progress variables
        self.training_progress = tk.DoubleVar(value=0.0)
        self.training_epoch = tk.IntVar(value=0)
        self.training_total_epochs = tk.IntVar(value=0)

        # Setup UI
        self._setup_ui()

        # Bind spacebar key
        self.root.bind('<space>', self._toggle_recording)

        # Start in run mode by default
        self._select_run_mode()

    def _set_custom_style(self):
        """Set custom theme and styling for the UI."""
        style = ttk.Style()

        # Use a modern theme as base
        if 'clam' in style.theme_names():
            style.theme_use('clam')

        # Configure colors
        style.configure('TFrame', background='#f5f5f5')
        style.configure('TLabel', background='#f5f5f5', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10, 'bold'), padding=5)
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
        style.configure('Recording.TLabel', foreground='red',
                        font=('Arial', 10, 'bold'))

        # Configure progress bar
        style.configure("green.Horizontal.TProgressbar",
                        troughcolor='#f0f0f0',
                        background='#4CAF50',
                        borderwidth=0,
                        thickness=10)

    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            messagebox.showerror("Error", f"Failed to load configuration: {e}")
            return {}

    def _load_available_gestures(self):
        """Load the list of available gestures from the dataset directory."""
        gestures = []
        dataset_path = self.config.get('data_collection', {}).get(
            'dataset_path', 'data/gesture_dataset')

        if os.path.exists(dataset_path):
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path) and not item.startswith('.') and item != 'examples':
                    sample_count = len(
                        [f for f in os.listdir(item_path) if f.endswith('.npy')])
                    gestures.append((item, sample_count))

        return sorted(gestures)

    def _setup_ui(self):
        """Setup the user interface."""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create left and right frames
        left_frame = ttk.Frame(main_frame, padding=5, width=800)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(main_frame, padding=5, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)

        # --- Left frame (video feed) ---
        video_frame = ttk.LabelFrame(left_frame, text="Camera Feed", padding=5)
        video_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Add a camera status indicator
        self.camera_status = ttk.Label(
            video_frame, text="Camera: Not started", foreground="gray")
        self.camera_status.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # --- Right frame (controls) ---
        # Mode selection with tabbed interface
        self.tab_control = ttk.Notebook(right_frame)

        # Create tabs
        self.collect_tab = ttk.Frame(self.tab_control, padding=10)
        self.train_tab = ttk.Frame(self.tab_control, padding=10)
        self.run_tab = ttk.Frame(self.tab_control, padding=10)

        self.tab_control.add(self.collect_tab, text="Collect Data")
        self.tab_control.add(self.train_tab, text="Train Model")
        self.tab_control.add(self.run_tab, text="Run Detection")

        self.tab_control.pack(fill=tk.BOTH, expand=True, pady=5)

        # Bind tab change event
        self.tab_control.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Quick access mode buttons
        mode_frame = ttk.Frame(right_frame, padding=5)
        mode_frame.pack(fill=tk.X, pady=5)

        # Collect mode options (tab content)
        # Create input frame
        collect_input_frame = ttk.Frame(self.collect_tab)
        collect_input_frame.pack(fill=tk.X, pady=5)

        # Gesture selection
        ttk.Label(collect_input_frame, text="Gesture Name:").grid(
            row=0, column=0, sticky=tk.W, pady=5)

        # Combo box for existing gestures + option to add new
        self.gesture_combo = ttk.Combobox(
            collect_input_frame, textvariable=self.current_gesture)
        self.gesture_combo.grid(
            row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        # We'll update this later after all UI components are created

        # Number of frames to collect
        ttk.Label(collect_input_frame, text="Frames to Collect:").grid(
            row=1, column=0, sticky=tk.W, pady=5)
        self.frames_entry = ttk.Entry(collect_input_frame)
        self.frames_entry.insert(0, str(self.config.get(
            'data_collection', {}).get('frames_per_gesture', 50)))
        self.frames_entry.grid(
            row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)

        # Add refresh button for gestures
        ttk.Button(collect_input_frame, text="Refresh", command=self._update_gesture_combo).grid(
            row=0, column=2, pady=5, padx=5)

        # Collection controls
        collect_controls_frame = ttk.Frame(self.collect_tab)
        collect_controls_frame.pack(fill=tk.X, pady=10)

        ttk.Button(collect_controls_frame, text="Start Collection",
                   command=self._start_collection).pack(side=tk.LEFT, padx=5)

        ttk.Button(collect_controls_frame, text="Export Dataset",
                   command=self._export_dataset).pack(side=tk.LEFT, padx=5)

        ttk.Button(collect_controls_frame, text="Import Dataset",
                   command=self._import_dataset).pack(side=tk.LEFT, padx=5)

        # Available gestures list with sample counts
        ttk.Label(self.collect_tab, text="Available Gestures:",
                  font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))

        # Create treeview for gestures
        gesture_columns = ('name', 'samples')
        self.gesture_tree = ttk.Treeview(
            self.collect_tab, columns=gesture_columns, show='headings', height=8)
        self.gesture_tree.heading('name', text='Gesture')
        self.gesture_tree.heading('samples', text='Samples')
        self.gesture_tree.column('name', width=150)
        self.gesture_tree.column('samples', width=80)
        self.gesture_tree.pack(fill=tk.BOTH, expand=True, pady=5)

        # Add right-click menu to gesture tree
        self.gesture_tree_menu = tk.Menu(self.gesture_tree, tearoff=0)
        self.gesture_tree_menu.add_command(
            label="Delete Gesture", command=self._delete_selected_gesture)
        self.gesture_tree_menu.add_command(
            label="Rename Gesture", command=self._rename_selected_gesture)

        self.gesture_tree.bind("<Button-3>", self._show_gesture_tree_menu)

        # Collection status frame
        collect_status_frame = ttk.Frame(self.collect_tab)
        collect_status_frame.pack(fill=tk.X, pady=5)

        self.collection_progress = ttk.Progressbar(
            collect_status_frame, style="green.Horizontal.TProgressbar", length=200)
        self.collection_progress.pack(fill=tk.X, pady=5)

        # Instructions
        instructions_frame = ttk.LabelFrame(
            self.collect_tab, text="Instructions")
        instructions_frame.pack(fill=tk.X, pady=5)

        instructions_text = (
            "1. Select or enter a gesture name\n"
            "2. Set the number of frames to collect\n"
            "3. Click 'Start Collection' and position yourself in the camera view\n"
            "4. Wait for the 5-second countdown to complete\n"
            "5. Hold the gesture steady until recording completes automatically"
        )

        ttk.Label(instructions_frame, text=instructions_text,
                  justify=tk.LEFT).pack(pady=5, padx=5)

        # Train mode options (tab content)
        train_input_frame = ttk.Frame(self.train_tab)
        train_input_frame.pack(fill=tk.X, pady=5)

        ttk.Label(train_input_frame, text="Dataset Path:").grid(
            row=0, column=0, sticky=tk.W, pady=5)
        self.dataset_path_entry = ttk.Entry(train_input_frame)
        self.dataset_path_entry.insert(0, self.config.get('data_collection', {}).get(
            'dataset_path', 'data/gesture_dataset'))
        self.dataset_path_entry.grid(
            row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(train_input_frame, text="Browse",
                   command=lambda: self._browse_directory(self.dataset_path_entry)).grid(row=0, column=2, pady=5)

        ttk.Label(train_input_frame, text="Model Output:").grid(
            row=1, column=0, sticky=tk.W, pady=5)
        self.model_path_entry = ttk.Entry(train_input_frame)
        self.model_path_entry.insert(0, self.config.get('model', {}).get(
            'model_path', 'models/gesture_model.h5'))
        self.model_path_entry.grid(
            row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(train_input_frame, text="Browse",
                   command=lambda: self._browse_file(self.model_path_entry)).grid(row=1, column=2, pady=5)

        # Advanced training options
        train_advanced_frame = ttk.LabelFrame(
            self.train_tab, text="Advanced Options")
        train_advanced_frame.pack(fill=tk.X, pady=5)

        # Number of epochs
        ttk.Label(train_advanced_frame, text="Epochs:").grid(
            row=0, column=0, sticky=tk.W, pady=5)
        self.epochs_entry = ttk.Entry(train_advanced_frame, width=8)
        self.epochs_entry.insert(0, "50")
        self.epochs_entry.grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)

        # Batch size
        ttk.Label(train_advanced_frame, text="Batch Size:").grid(
            row=0, column=2, sticky=tk.W, pady=5)
        self.batch_size_entry = ttk.Entry(train_advanced_frame, width=8)
        self.batch_size_entry.insert(0, "32")
        self.batch_size_entry.grid(
            row=0, column=3, sticky=tk.W, pady=5, padx=5)

        # Validation split
        ttk.Label(train_advanced_frame, text="Validation Split:").grid(
            row=1, column=0, sticky=tk.W, pady=5)
        self.val_split_entry = ttk.Entry(train_advanced_frame, width=8)
        self.val_split_entry.insert(0, "0.2")
        self.val_split_entry.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)

        # Learning rate
        ttk.Label(train_advanced_frame, text="Learning Rate:").grid(
            row=1, column=2, sticky=tk.W, pady=5)
        self.learning_rate_entry = ttk.Entry(train_advanced_frame, width=8)
        self.learning_rate_entry.insert(0, "0.001")
        self.learning_rate_entry.grid(
            row=1, column=3, sticky=tk.W, pady=5, padx=5)

        # Training controls
        train_controls_frame = ttk.Frame(self.train_tab)
        train_controls_frame.pack(fill=tk.X, pady=10)

        ttk.Button(train_controls_frame, text="Start Training",
                   command=self._start_training).pack(side=tk.LEFT, padx=5)

        ttk.Button(train_controls_frame, text="Test Model",
                   command=self._test_model).pack(side=tk.LEFT, padx=5)

        ttk.Button(train_controls_frame, text="Reset Model",
                   command=self._reset_model).pack(side=tk.LEFT, padx=5)

        # Training progress
        train_progress_frame = ttk.LabelFrame(
            self.train_tab, text="Training Progress")
        train_progress_frame.pack(fill=tk.X, pady=5)

        self.train_progress_bar = ttk.Progressbar(
            train_progress_frame, style="green.Horizontal.TProgressbar",
            length=200, variable=self.training_progress)
        self.train_progress_bar.pack(fill=tk.X, pady=5)

        self.train_status_label = ttk.Label(
            train_progress_frame, text="Ready to train")
        self.train_status_label.pack(fill=tk.X, pady=5)

        # Training log
        train_log_frame = ttk.LabelFrame(self.train_tab, text="Training Log")
        train_log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.train_log = scrolledtext.ScrolledText(
            train_log_frame, height=6, wrap=tk.WORD)
        self.train_log.pack(fill=tk.BOTH, expand=True, pady=5)
        self.train_log.config(state=tk.DISABLED)

        # Run mode options (tab content)
        run_input_frame = ttk.Frame(self.run_tab)
        run_input_frame.pack(fill=tk.X, pady=5)

        ttk.Label(run_input_frame, text="Model Path:").grid(
            row=0, column=0, sticky=tk.W, pady=5)
        self.run_model_path = ttk.Entry(run_input_frame)
        self.run_model_path.insert(0, self.config.get('model', {}).get(
            'model_path', 'models/gesture_model.h5'))
        self.run_model_path.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(run_input_frame, text="Browse",
                   command=lambda: self._browse_file(self.run_model_path)).grid(row=0, column=2, pady=5)

        # Confidence threshold
        ttk.Label(run_input_frame, text="Confidence Threshold:").grid(
            row=1, column=0, sticky=tk.W, pady=5)

        confidence_frame = ttk.Frame(run_input_frame)
        confidence_frame.grid(row=1, column=1, columnspan=2,
                              sticky=(tk.W, tk.E), pady=5)

        self.confidence_scale = ttk.Scale(
            confidence_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL)
        self.confidence_scale.set(self.config.get('detection', {}).get(
            'confidence_threshold', 0.7))
        self.confidence_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.confidence_label = ttk.Label(confidence_frame, text="0.7")
        self.confidence_label.pack(side=tk.RIGHT, padx=5)

        # Update label when scale changes
        self.confidence_scale.configure(command=self._update_confidence_label)

        # Run controls
        run_controls_frame = ttk.Frame(self.run_tab)
        run_controls_frame.pack(fill=tk.X, pady=10)

        ttk.Button(run_controls_frame, text="Start Detection",
                   command=self._start_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(run_controls_frame, text="Stop Detection",

                   command=self._stop_detection).pack(side=tk.LEFT, padx=5)

        ttk.Button(run_controls_frame, text="Edit Triggers",
                   command=self._edit_triggers).pack(side=tk.LEFT, padx=5)

        # Create trigger display
        trigger_frame = ttk.LabelFrame(self.run_tab, text="Active Triggers")
        trigger_frame.pack(fill=tk.X, pady=5)

        # Create treeview for triggers
        trigger_columns = ('gesture', 'action', 'details')
        self.trigger_tree = ttk.Treeview(
            trigger_frame, columns=trigger_columns, show='headings', height=5)
        self.trigger_tree.heading('gesture', text='Gesture')
        self.trigger_tree.heading('action', text='Action')
        self.trigger_tree.heading('details', text='Details')
        self.trigger_tree.column('gesture', width=100)
        self.trigger_tree.column('action', width=100)
        self.trigger_tree.column('details', width=150)
        self.trigger_tree.pack(fill=tk.BOTH, expand=True, pady=5)

        # Load triggers
        self._load_triggers()

        # Detected gestures frame
        gestures_frame = ttk.LabelFrame(self.run_tab, text="Detected Gestures")
        gestures_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create treeview for detected gestures
        gesture_columns = ('name', 'confidence', 'timestamp')
        self.gesture_list = ttk.Treeview(
            gestures_frame, columns=gesture_columns, show='headings', height=8)
        self.gesture_list.heading('name', text='Gesture')
        self.gesture_list.heading('confidence', text='Confidence')
        self.gesture_list.heading('timestamp', text='Time')

        # Center align all columns for better readability
        self.gesture_list.column('name', width=100, anchor='center')
        self.gesture_list.column('confidence', width=100, anchor='center')
        self.gesture_list.column('timestamp', width=100, anchor='center')

        # Add scrollbars for better navigation
        gesture_y_scrollbar = ttk.Scrollbar(
            gestures_frame, orient="vertical", command=self.gesture_list.yview)
        gesture_x_scrollbar = ttk.Scrollbar(
            gestures_frame, orient="horizontal", command=self.gesture_list.xview)

        self.gesture_list.configure(
            yscrollcommand=gesture_y_scrollbar.set, xscrollcommand=gesture_x_scrollbar.set)

        # Configure the gesture list appearance and tags
        # Light green background for triggered gestures
        self.gesture_list.tag_configure('triggered', background='#e6ffe6')

        # Pack scrollbars and treeview
        gesture_y_scrollbar.pack(side='right', fill='y')
        gesture_x_scrollbar.pack(side='bottom', fill='x')
        self.gesture_list.pack(fill=tk.BOTH, expand=True, pady=5)

        # Detection log
        detection_log_frame = ttk.LabelFrame(
            self.run_tab, text="Detection Log")
        detection_log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Configure detection log with proper font, colors and scrollbar
        self.detection_log = scrolledtext.ScrolledText(
            detection_log_frame, height=8, wrap=tk.WORD,
            font=('Consolas', 10), background='#f8f8f8')

        # Add tag configurations for different message types
        self.detection_log.tag_configure('detected', foreground='blue')
        self.detection_log.tag_configure('triggered', foreground='green')
        self.detection_log.tag_configure('error', foreground='red')
        self.detection_log.tag_configure('info', foreground='black')

        # Add horizontal scrollbar
        detection_x_scrollbar = ttk.Scrollbar(
            detection_log_frame, orient="horizontal", command=self.detection_log.xview)
        self.detection_log.configure(xscrollcommand=detection_x_scrollbar.set)

        # Pack scrollbar and text widget
        detection_x_scrollbar.pack(side='bottom', fill='x')
        self.detection_log.pack(fill=tk.BOTH, expand=True, pady=5)

        # Disable editing
        self.detection_log.config(state=tk.DISABLED)

        # Add status frame (common to all modes)
        status_frame = ttk.LabelFrame(right_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=5, side=tk.BOTTOM)

        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(fill=tk.X)

        # Control buttons
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(fill=tk.X, pady=10, before=status_frame)

        self.start_stop_button = ttk.Button(
            control_frame, text="Start", command=self._toggle_running)
        self.start_stop_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Exit", command=self._exit_application).pack(
            side=tk.RIGHT, padx=5)

    def _on_tab_changed(self, event):
        """Handle tab change events."""
        current_tab = self.tab_control.index(self.tab_control.select())

        # Stop any running processes when switching tabs
        self._stop_running()

        # Update mode based on selected tab
        if current_tab == 0:  # Collect tab
            self.current_mode.set("collect")
            self._select_collect_mode()
        elif current_tab == 1:  # Train tab
            self.current_mode.set("train")
            self._select_train_mode()
        elif current_tab == 2:  # Run tab
            self.current_mode.set("run")
            self._select_run_mode()

    def _select_collect_mode(self):
        """Switch to data collection mode."""
        self._stop_running()
        # Update available gestures
        self._update_gesture_combo()
        # Select the collect tab
        self.tab_control.select(0)
        self.status_label.config(
            text="Ready to collect data. Enter a gesture name and press Start.")

    def _select_train_mode(self):
        """Switch to model training mode."""
        self._stop_running()
        # Select the train tab
        self.tab_control.select(1)
        self.status_label.config(
            text="Ready to train model. Configure options and press Start.")

    def _select_run_mode(self):
        """Switch to detection/run mode."""
        self._stop_running()
        # Select the run tab
        self.tab_control.select(2)
        # Load triggers
        self._load_triggers()
        self.status_label.config(text="Ready to detect gestures. Press Start.")

    def _browse_directory(self, entry_widget):
        """Open directory browser and update entry widget."""
        directory = filedialog.askdirectory()
        if directory:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, directory)

    def _browse_file(self, entry_widget):
        """Open file browser and update entry widget."""
        file_path = filedialog.askopenfilename()
        if file_path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, file_path)

    def _toggle_running(self):
        """Toggle between running and stopped states."""
        if self.running:
            self._stop_running()
        else:
            if self.current_mode.get() == "collect":
                self._start_collection()
            elif self.current_mode.get() == "train":
                self._start_training()
            elif self.current_mode.get() == "run":
                self._start_detection()

    def _toggle_recording(self, event=None):
        """Toggle the recording state."""
        self.recording = not self.recording
        if self.recording:
            self.status_label.config(
                text="Recording started. Press SPACE to stop.")
        else:
            self.status_label.config(
                text="Recording stopped. Press SPACE to start.")

    def _start_collection(self):
        """Start collecting data for the specified gesture."""
        gesture_name = self.current_gesture.get().strip()
        if not gesture_name:
            messagebox.showerror("Error", "Please enter a gesture name")
            return

        # Make sure any previous collection is stopped
        if self.running:
            self._stop_running()
            # Wait a moment for resources to be cleaned up
            time.sleep(0.5)

        # Reset collection state
        self.running = True
        self.recording = False
        self.collected_frames = 0
        self.countdown_active = True  # Add a flag for the countdown
        self.countdown_seconds = 5    # Set the countdown duration
        self.start_stop_button.config(text="Stop")
        self.status_label.config(
            text=f"Get ready! Countdown: {self.countdown_seconds} seconds")

        # Start collection in a separate thread
        self.stop_event.clear()
        self.video_thread = threading.Thread(
            target=self._collection_thread, args=(gesture_name,))
        self.video_thread.daemon = True
        self.video_thread.start()

    def _collection_thread(self, gesture_name):
        """Background thread for data collection."""
        try:
            collector = DataCollector(self.config)

            # Use camera settings from config.yaml
            camera_config = self.config.get('camera', {})
            device_id = camera_config.get('device_id', 0)
            width = camera_config.get('width', 640)
            height = camera_config.get('height', 480)
            fps = camera_config.get('fps', 30)

            logger.info(
                f"Using camera settings: Device ID={device_id}, Resolution={width}x{height}, FPS={fps}")

            cap = cv2.VideoCapture(device_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)

            if not cap.isOpened():
                logger.error(f"Failed to open camera (Device ID: {device_id})")
                messagebox.showerror(
                    "Error", f"Failed to open camera (Device ID: {device_id})")
                self._stop_running()
                return

            # Create gesture directory
            gesture_dir = os.path.join(self.config.get('data_collection', {}).get(
                'dataset_path', 'data/gesture_dataset'), gesture_name)
            os.makedirs(gesture_dir, exist_ok=True)

            # Count existing samples
            existing_samples = len(
                [f for f in os.listdir(gesture_dir) if f.endswith('.npy')])

            self.frames_to_collect = self.config.get(
                'data_collection', {}).get('frames_per_gesture', 50)
            self.collected_frames = 0

            # Initialize countdown variables
            self.countdown_time = time.time()
            self.countdown_active = True

            # Main collection loop
            while not self.stop_event.is_set():
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Mirror the frame for intuitive feedback
                frame = cv2.flip(frame, 1)

                # Handle countdown before recording
                if self.countdown_active:
                    elapsed_time = time.time() - self.countdown_time
                    remaining_seconds = max(
                        0, int(self.countdown_seconds - elapsed_time))

                    # Update UI with countdown
                    countdown_text = f"Get ready! Starting in {remaining_seconds}..."
                    self.status_label.config(text=countdown_text)

                    # Display countdown on frame
                    cv2.putText(frame, countdown_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Check if countdown finished
                    if remaining_seconds == 0:
                        self.countdown_active = False
                        self.recording = True
                        self.status_label.config(
                            text=f"Recording {gesture_name}: 0/{self.frames_to_collect}")

                elif self.recording:
                    # Extract features
                    features = collector._extract_features(frame)

                    # Save features
                    sample_path = os.path.join(
                        gesture_dir, f"{gesture_name}_{existing_samples + self.collected_frames}.npy")
                    np.save(sample_path, features)

                    self.collected_frames += 1

                    # Update status
                    self.status_label.config(
                        text=f"Recording {gesture_name}: {self.collected_frames}/{self.frames_to_collect}")

                    # Draw recording indicator
                    cv2.putText(frame, f"Recording: {self.collected_frames}/{self.frames_to_collect}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Check if done
                    if self.collected_frames >= self.frames_to_collect:
                        self.recording = False
                        self.status_label.config(
                            text=f"Completed! Collected {self.frames_to_collect} frames for {gesture_name}")

                        # Exit the collection loop and reset for a new collection
                        # Wait a moment to show the completion message
                        time.sleep(1.5)
                        # Signal the thread to stop cleanly
                        self.stop_event.set()
                        break
                else:
                    # Only show this when we're not in countdown mode and not recording
                    if not self.countdown_active:
                        # Show completion message or idle state
                        cv2.putText(frame, "Collection paused or complete", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Convert frame to format for tkinter
                cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cv_image)
                tk_image = ImageTk.PhotoImage(image=pil_image)

                # Update video display
                self.video_label.config(image=tk_image)
                self.video_label.image = tk_image

                # Small delay to reduce CPU usage
                time.sleep(0.01)

            # Cleanup
            cap.release()

        except Exception as e:
            logger.error(f"Error in collection thread: {e}")
            messagebox.showerror("Error", f"Collection failed: {e}")
        finally:
            self._stop_running()

    def _start_training(self):
        """Start training the model with collected data."""
        self.running = True
        self.start_stop_button.config(text="Stop")
        self.status_label.config(text="Training model...")

        # Start training in a separate thread
        self.stop_event.clear()
        self.video_thread = threading.Thread(target=self._training_thread)
        self.video_thread.daemon = True
        self.video_thread.start()

    def _training_thread(self):
        """Background thread for model training."""
        try:
            # Create a queue for communication between threads
            self.training_queue = queue.Queue()

            # Create an event to signal stopping to TensorFlow
            self.tf_stop_event = threading.Event()

            # Initialize training progress
            self.training_progress.set(0.0)
            self.train_status_label.config(text="Preparing to train...")

            # Clear the training log
            self.train_log.config(state=tk.NORMAL)
            self.train_log.delete(1.0, tk.END)
            self.train_log.config(state=tk.DISABLED)

            # Start a thread for updating the UI based on training progress
            def ui_updater():
                while self.running and not self.stop_event.is_set():
                    try:
                        # Non-blocking get to avoid freezing
                        progress_data = self.training_queue.get(
                            block=True, timeout=0.5)

                        # Update UI based on progress data
                        if 'progress_percent' in progress_data:
                            # Update progress bar
                            self.training_progress.set(
                                progress_data['progress_percent'])

                            # Update status label
                            epoch = progress_data['epoch']
                            total = progress_data['total_epochs']
                            acc = progress_data.get('accuracy', 0) * 100
                            val_acc = progress_data.get(
                                'val_accuracy', 0) * 100

                            status_text = f"Training: Epoch {epoch}/{total} - Accuracy: {acc:.2f}% - Val Accuracy: {val_acc:.2f}%"
                            self.train_status_label.config(text=status_text)

                            # Update log
                            log_text = f"Epoch {epoch}/{total}: accuracy={acc:.2f}%, val_accuracy={val_acc:.2f}%\n"
                            self.train_log.config(state=tk.NORMAL)
                            self.train_log.insert(tk.END, log_text)
                            self.train_log.see(tk.END)
                            self.train_log.config(state=tk.DISABLED)

                        elif 'status' in progress_data and progress_data['status'] == 'complete':
                            # Training complete, update UI
                            self.train_status_label.config(
                                text="Training complete!")
                            self.training_progress.set(100.0)
                            break

                    except queue.Empty:
                        # No progress data, continue waiting
                        continue
                    except Exception as e:
                        logger.error(f"Error in UI updater: {e}")
                        break

            # Start UI updater thread
            ui_thread = threading.Thread(target=ui_updater)
            ui_thread.daemon = True
            ui_thread.start()

            # Import here to avoid circular imports
            from src.gui_training_helpers import TrainingCallback

            # Configure the trainer
            trainer = ModelTrainer(self.config)

            # Start a detached thread for the actual training
            def do_training():
                try:
                    # Load dataset
                    self.training_queue.put({'status': 'loading_data'})
                    X, y, label_map = trainer._load_dataset()

                    if X is None or y is None or len(X) == 0:
                        # Dataset loading failed or is empty
                        self.root.after(0, lambda: messagebox.showerror(
                            "Error", "No training data found. Please collect gesture data first."))
                        return

                    # Get parameters from UI
                    try:
                        epochs = int(self.epochs_entry.get())
                        batch_size = int(self.batch_size_entry.get())
                        learning_rate = float(self.learning_rate_entry.get())
                    except ValueError:
                        # Default values if parsing fails
                        epochs = 50
                        batch_size = 32
                        learning_rate = 0.001

                    # Train model with custom callback for progress updates
                    custom_callback = TrainingCallback(
                        self.training_queue, self.tf_stop_event)
                    model, label_map = trainer.train_model(epochs=epochs,
                                                           batch_size=batch_size,
                                                           learning_rate=learning_rate,
                                                           custom_callback=custom_callback)

                    # Check if training was canceled
                    if self.stop_event.is_set() or self.tf_stop_event.is_set():
                        return

                    # Notify success on the main thread
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Success", "Model training completed successfully!"))

                except Exception as e:
                    logger.error(f"Error during training: {e}")
                    # Show error on the main thread
                    # Copy the error message to avoid referencing e in the lambda
                    error_message = str(e)
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", f"Training failed: {error_message}"))
                finally:
                    # Ensure we signal completion
                    self.training_queue.put({'status': 'complete'})
                    # Clean up on the main thread
                    self.root.after(0, self._stop_running)

            # Start the actual training in a separate thread
            training_thread = threading.Thread(target=do_training)
            training_thread.daemon = True
            training_thread.start()

        except Exception as e:
            logger.error(f"Error setting up training: {e}")
            # Use root.after to show message on main thread
            self.root.after(0, lambda: messagebox.showerror(
                "Error", f"Failed to setup training: {e}"))
            self._stop_running()

    def _start_detection(self):
        """Start gesture detection and trigger execution."""
        self.running = True
        self.start_stop_button.config(text="Stop")
        self.status_label.config(text="Starting gesture detection...")

        # Get the current confidence threshold from the slider
        current_threshold = float(self.confidence_scale.get())
        self.config['detection']['confidence_threshold'] = current_threshold

        # Clear the gesture list
        for item in self.gesture_list.get_children():
            self.gesture_list.delete(item)

        # Start detection in a separate thread
        self.stop_event.clear()
        self.video_thread = threading.Thread(target=self._detection_thread)
        self.video_thread.daemon = True
        self.video_thread.start()

    def _stop_detection(self):
        """Stop gesture detection and reset UI."""
        if self.running and self.current_mode.get() == "run":
            logger.info("Stopping detection gracefully...")
            # Set the status before stopping to avoid race conditions
            self.status_label.config(text="Stopping detection...")
            # Stop the detection thread and clean up resources
            self._stop_running()
            self.status_label.config(text="Detection stopped.")

    def _detection_thread(self):
        """Background thread for gesture detection."""
        try:
            # Create a trigger log manager to handle UI updates
            from src.gui_trigger_helpers import TriggerLogManager
            self.trigger_log_manager = TriggerLogManager(
                detection_log=self.detection_log,
                gesture_list=self.gesture_list
            )

            # Initialize components
            self.detector = GestureDetector(self.config)

            # Make sure the detector has the latest confidence threshold
            detection_config = self.config.get('detection', {})
            confidence_threshold = detection_config.get(
                'confidence_threshold', 0.7)
            self.detector.detection_config['confidence_threshold'] = confidence_threshold
            logger.info(f"Using confidence threshold: {confidence_threshold}")

            # Initialize the trigger manager with our log manager
            self.trigger_manager = TriggerManager(
                self.config,
                trigger_log_manager=self.trigger_log_manager
            )

            # Start the detector
            self.detector.start()
            self.status_label.config(text="Detecting gestures...")

            # Clear detection log and gesture list
            self.trigger_log_manager.clear_logs()

            # Main detection loop
            while not self.stop_event.is_set() and self.running:
                # Check if detector is still valid
                if not self.detector:
                    break

                try:
                    # Get the current frame
                    frame = self.detector.get_frame()

                    if frame is not None:
                        # Convert frame to format for tkinter
                        cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(cv_image)
                        tk_image = ImageTk.PhotoImage(image=pil_image)

                        # Update video display
                        self.video_label.config(image=tk_image)
                        self.video_label.image = tk_image

                    # Get detected gestures
                    gestures = self.detector.detect()

                    # Update gesture list
                    self.gesture_list.delete(*self.gesture_list.get_children())

                    # Process detected gestures
                    for detected_gesture, confidence in gestures:
                        # Get the current confidence threshold from config
                        confidence_threshold = self.config.get(
                            'detection', {}).get('confidence_threshold', 0.7)

                        # Log all detected gestures to the UI
                        self.trigger_log_manager.log_detected_gesture(
                            detected_gesture, confidence)

                        # Trigger action only if confidence exceeds threshold
                        if confidence > confidence_threshold:
                            # Pass confidence value to the trigger manager
                            self.trigger_manager.execute_trigger(
                                detected_gesture, confidence=confidence)
                except Exception as e:
                    if not self.stop_event.is_set():
                        # Only log errors if we're not intentionally stopping
                        logger.error(f"Error in detection loop: {e}")
                    break

                # Small delay to reduce CPU usage
                time.sleep(0.01)

        except Exception as e:
            # Only show the error message if we're not in the process of stopping
            if not self.stop_event.is_set():
                logger.error(f"Error in detection thread: {e}")
                messagebox.showerror("Error", f"Detection failed: {e}")
        finally:
            if self.detector:
                self.detector.stop()
            if self.trigger_manager:
                self.trigger_manager.cleanup()
            self._stop_running()

    def _stop_running(self):
        """Stop the current operation and clean up."""
        # First, signal the thread to stop
        self.stop_event.set()
        self.running = False
        self.recording = False
        self.countdown_active = False
        self.start_stop_button.config(text="Start")

        # Signal TensorFlow to stop, if training
        if hasattr(self, 'tf_stop_event') and self.tf_stop_event is not None:
            self.tf_stop_event.set()

        # Only join the thread if it's not the current thread
        current_thread = threading.current_thread()
        if self.video_thread and self.video_thread.is_alive() and self.video_thread is not current_thread:
            self.video_thread.join(timeout=1.0)

        # Now it's safe to clean up resources
        if self.detector:
            self.detector.stop()
            self.detector = None

        if self.trigger_manager:
            self.trigger_manager.cleanup()
            self.trigger_manager = None

    def _exit_application(self):
        """Exit the application cleanly."""
        self._stop_running()
        self.root.quit()

    def run(self):
        """Run the main application loop."""
        # Now that all UI components are created, update the gesture combo
        self._update_gesture_combo()

        self.root.protocol("WM_DELETE_WINDOW", self._exit_application)
        self.root.mainloop()

    def _update_gesture_combo(self):
        """Update the gesture combobox with available gestures."""
        # Refresh available gestures
        self.available_gestures = self._load_available_gestures()

        # Update combobox if it exists
        if hasattr(self, 'gesture_combo'):
            # Clear existing items in combobox
            self.gesture_combo['values'] = []

            # Add available gesture names to combobox
            if self.available_gestures:
                gesture_names = [g[0] for g in self.available_gestures]
                self.gesture_combo['values'] = gesture_names

        # Update gesture tree
        self._update_gesture_tree()

    def _update_gesture_tree(self):
        """Update the gesture treeview with available gestures and their sample counts."""
        # Ensure gesture_tree exists before attempting to update it
        if not hasattr(self, 'gesture_tree'):
            return

        # Clear existing items
        for item in self.gesture_tree.get_children():
            self.gesture_tree.delete(item)

        # Add gestures and their sample counts
        for gesture, sample_count in self.available_gestures:
            self.gesture_tree.insert(
                '', 'end', text=gesture, values=(gesture, sample_count))

    def _show_gesture_tree_menu(self, event):
        """Show context menu for gesture tree."""
        # Get the item under cursor
        item = self.gesture_tree.identify_row(event.y)
        if item:
            # Select the item
            self.gesture_tree.selection_set(item)
            # Show popup menu
            self.gesture_tree_menu.post(event.x_root, event.y_root)

    def _delete_selected_gesture(self):
        """Delete the selected gesture from the dataset."""
        selected = self.gesture_tree.selection()
        if not selected:
            return

        item = selected[0]
        gesture = self.gesture_tree.item(item, 'values')[0]

        # Confirm deletion
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete all samples for '{gesture}'?"):
            try:
                # Delete the gesture directory
                gesture_dir = os.path.join(self.config.get('data_collection', {}).get(
                    'dataset_path', 'data/gesture_dataset'), gesture)
                if os.path.exists(gesture_dir):
                    shutil.rmtree(gesture_dir)
                    messagebox.showinfo(
                        "Success", f"Deleted gesture: {gesture}")

                    # Update UI
                    self._update_gesture_combo()
                else:
                    messagebox.showerror(
                        "Error", f"Gesture directory not found: {gesture_dir}")
            except Exception as e:
                logger.error(f"Error deleting gesture: {e}")
                messagebox.showerror("Error", f"Failed to delete gesture: {e}")

    def _rename_selected_gesture(self):
        """Rename the selected gesture."""
        selected = self.gesture_tree.selection()
        if not selected:
            return

        item = selected[0]
        old_name = self.gesture_tree.item(item, 'values')[0]

        # Ask for new name
        new_name = simpledialog.askstring("Rename Gesture",
                                          f"Enter new name for '{old_name}':",
                                          parent=self.root)

        if not new_name:
            return

        # Validate new name
        if new_name == old_name:
            return

        if not new_name.isalnum() and not '_' in new_name:
            messagebox.showerror(
                "Error", "Gesture name can only contain letters, numbers, and underscores.")
            return

        try:
            # Rename the gesture directory
            old_dir = os.path.join(self.config.get('data_collection', {}).get(
                'dataset_path', 'data/gesture_dataset'), old_name)
            new_dir = os.path.join(self.config.get('data_collection', {}).get(
                'dataset_path', 'data/gesture_dataset'), new_name)

            if os.path.exists(new_dir):
                messagebox.showerror(
                    "Error", f"A gesture named '{new_name}' already exists.")
                return

            if os.path.exists(old_dir):
                # Create new directory
                os.makedirs(new_dir, exist_ok=True)

                # Copy all files with renamed filenames
                for file in os.listdir(old_dir):
                    if file.endswith('.npy'):
                        # Rename the file to use the new gesture name
                        new_file = file.replace(old_name, new_name)
                        shutil.copy2(os.path.join(old_dir, file),
                                     os.path.join(new_dir, new_file))

                # Remove old directory
                shutil.rmtree(old_dir)

                messagebox.showinfo(
                    "Success", f"Renamed gesture: {old_name} → {new_name}")

                # Update UI
                self._update_gesture_combo()
            else:
                messagebox.showerror(
                    "Error", f"Gesture directory not found: {old_dir}")
        except Exception as e:
            logger.error(f"Error renaming gesture: {e}")
            messagebox.showerror("Error", f"Failed to rename gesture: {e}")

    def _export_dataset(self):
        """Export the gesture dataset."""
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return

        try:
            # Create a timestamped zip file
            dataset_path = self.config.get('data_collection', {}).get(
                'dataset_path', 'data/gesture_dataset')

            if not os.path.exists(dataset_path):
                messagebox.showerror(
                    "Error", "Dataset directory does not exist.")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = os.path.join(
                export_dir, f"gesture_dataset_{timestamp}.zip")

            # Create export info file
            info_file = os.path.join(dataset_path, "_export_info.txt")
            with open(info_file, 'w') as f:
                f.write("Gesture Dataset Export\n")
                f.write(
                    f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Gestures:\n")

                for gesture, sample_count in self.available_gestures:
                    f.write(f"- {gesture}: {sample_count} samples\n")

            # Create zip archive
            shutil.make_archive(zip_filename[:-4], 'zip', dataset_path)

            # Remove temporary info file
            os.remove(info_file)

            messagebox.showinfo(
                "Success", f"Dataset exported to:\n{zip_filename}")
        except Exception as e:
            logger.error(f"Error exporting dataset: {e}")
            messagebox.showerror("Error", f"Failed to export dataset: {e}")

    def _import_dataset(self):
        """Import a gesture dataset."""
        zip_file = filedialog.askopenfilename(
            title="Select Dataset ZIP File",
            filetypes=[("ZIP Files", "*.zip")])

        if not zip_file:
            return

        try:
            # Ask for confirmation if dataset already has gestures
            if self.available_gestures:
                choice = messagebox.askyesnocancel(
                    "Import Options",
                    "How do you want to import the dataset?\n\n"
                    "Yes: Merge with existing dataset\n"
                    "No: Replace existing dataset\n"
                    "Cancel: Abort import")

                if choice is None:  # Cancel
                    return

                if not choice:  # No - replace
                    # Delete existing dataset
                    dataset_path = self.config.get('data_collection', {}).get(
                        'dataset_path', 'data/gesture_dataset')

                    # Keep examples directory if it exists
                    examples_dir = os.path.join(dataset_path, "examples")
                    if os.path.exists(examples_dir):
                        # Create a temporary directory for examples
                        temp_examples = os.path.join(
                            os.path.dirname(dataset_path), "_temp_examples")
                        shutil.copytree(examples_dir, temp_examples)

                    # Delete the dataset directory
                    if os.path.exists(dataset_path):
                        shutil.rmtree(dataset_path)

                    # Recreate the directory
                    os.makedirs(dataset_path, exist_ok=True)

                    # Restore examples
                    if os.path.exists(temp_examples):
                        shutil.copytree(temp_examples, examples_dir)
                        shutil.rmtree(temp_examples)

            # Extract the ZIP file
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract to temp directory
                shutil.unpack_archive(zip_file, temp_dir, 'zip')

                # Copy each gesture directory
                dataset_path = self.config.get('data_collection', {}).get(
                    'dataset_path', 'data/gesture_dataset')

                # Create dataset directory if it doesn't exist
                os.makedirs(dataset_path, exist_ok=True)

                # Get all gesture directories
                gesture_dirs = [d for d in os.listdir(temp_dir)
                                if os.path.isdir(os.path.join(temp_dir, d))
                                and not d.startswith('.')
                                and not d.startswith('_')
                                and d != 'examples']

                # Copy each gesture directory
                for gesture_dir in gesture_dirs:
                    src_dir = os.path.join(temp_dir, gesture_dir)
                    dst_dir = os.path.join(dataset_path, gesture_dir)

                    if os.path.exists(dst_dir):
                        # Merge directories
                        for file in os.listdir(src_dir):
                            if file.endswith('.npy'):
                                # Get current count of files
                                existing_count = len(
                                    [f for f in os.listdir(dst_dir) if f.endswith('.npy')])

                                # Create new filename with incremented index
                                new_file = f"{gesture_dir}_{existing_count}.npy"

                                # Copy file with new name
                                shutil.copy2(
                                    os.path.join(src_dir, file),
                                    os.path.join(dst_dir, new_file)
                                )
                    else:
                        # Just copy the directory
                        shutil.copytree(src_dir, dst_dir)

            # Update UI
            self._update_gesture_combo()

            # Show success message
            messagebox.showinfo("Success", "Dataset imported successfully!")
        except Exception as e:
            logger.error(f"Error importing dataset: {e}")
            messagebox.showerror("Error", f"Failed to import dataset: {e}")

    def _update_confidence_label(self, value):
        """Update confidence threshold label when slider is moved."""
        value = float(value)
        self.confidence_label.config(text=f"{value:.1f}")

        # Update the configuration value with the new threshold
        if 'detection' not in self.config:
            self.config['detection'] = {}
        self.config['detection']['confidence_threshold'] = value

        # Update detector's configuration if it exists
        if hasattr(self, 'detector') and self.detector is not None:
            self.detector.detection_config['confidence_threshold'] = value

    def _test_model(self):
        """Test the trained model with live camera feed."""
        if not os.path.exists(self.model_path_entry.get()):
            messagebox.showerror(
                "Error", "Model file not found. Please train a model first.")
            return

        # Switch to run mode to test
        self.tab_control.select(2)  # Select the run tab

        # Update model path in run tab
        self.run_model_path.delete(0, tk.END)
        self.run_model_path.insert(0, self.model_path_entry.get())

        # Start detection
        self._start_detection()

    def _reset_model(self):
        """Reset the model to start training from scratch."""
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset the model? This will delete the existing model file."):
            model_path = self.model_path_entry.get()

            if os.path.exists(model_path):
                try:
                    os.remove(model_path)

                    # Also remove the label map
                    label_map_path = os.path.join(
                        os.path.dirname(model_path), "label_map.npy")
                    if os.path.exists(label_map_path):
                        os.remove(label_map_path)

                    messagebox.showinfo("Success", "Model reset successfully.")
                except Exception as e:
                    logger.error(f"Error resetting model: {e}")
                    messagebox.showerror(
                        "Error", f"Failed to reset model: {e}")
            else:
                messagebox.showinfo("Info", "No existing model found.")

    def _load_triggers(self):
        """Load and display configured triggers."""
        # Clear the trigger tree
        for item in self.trigger_tree.get_children():
            self.trigger_tree.delete(item)

        try:
            # Load triggers from config
            trigger_config_path = self.config.get('triggers', {}).get(
                'trigger_config_path', 'config/triggers.yaml')

            if not os.path.exists(trigger_config_path):
                logger.warning(
                    f"Trigger config file not found: {trigger_config_path}")
                return

            with open(trigger_config_path, 'r') as f:
                trigger_config = yaml.safe_load(f)

            # Add triggers to tree
            if 'triggers' in trigger_config:
                for gesture, trigger_info in trigger_config['triggers'].items():
                    trigger_type = trigger_info.get('type', 'unknown')

                    # Get details based on trigger type
                    details = ""
                    if trigger_type == 'sound':
                        details = trigger_info.get(
                            'params', {}).get('sound_file', '')
                    elif trigger_type == 'visual':
                        effect = trigger_info.get(
                            'params', {}).get('effect', '')
                        color = trigger_info.get('params', {}).get('color', [])
                        details = f"{effect} ({','.join(map(str, color))})"
                    elif trigger_type == 'midi':
                        note = trigger_info.get('params', {}).get('note', '')
                        details = f"Note: {note}"
                    elif trigger_type == 'custom':
                        module = trigger_info.get('module', '')
                        function = trigger_info.get('function', '')
                        details = f"{module}.{function}"

                    # Add to tree
                    self.trigger_tree.insert('', 'end', text=gesture,
                                             values=(gesture, trigger_type, details))
        except Exception as e:
            logger.error(f"Error loading triggers: {e}")

    def _edit_triggers(self):
        """Open the triggers configuration file for editing."""
        trigger_config_path = self.config.get('triggers', {}).get(
            'trigger_config_path', 'config/triggers.yaml')

        if not os.path.exists(trigger_config_path):
            messagebox.showerror(
                "Error", f"Trigger config file not found: {trigger_config_path}")
            return

        # Open the file with the default system application
        import subprocess
        import platform

        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', trigger_config_path])
            elif platform.system() == 'Windows':
                subprocess.run(['start', trigger_config_path], shell=True)
            else:  # Linux
                subprocess.run(['xdg-open', trigger_config_path])

            messagebox.showinfo("Info",
                                "The trigger configuration file has been opened for editing.\n"
                                "Save the file after making changes, then restart the application to apply them.")
        except Exception as e:
            logger.error(f"Error opening trigger config: {e}")
            messagebox.showerror(
                "Error", f"Failed to open trigger config file: {e}")

        # Prompt to reload
        if messagebox.askyesno("Reload Triggers", "Would you like to reload the triggers after editing?"):
            self._load_triggers()


if __name__ == "__main__":
    app = GestureTriggerGUI()
    app.run()
