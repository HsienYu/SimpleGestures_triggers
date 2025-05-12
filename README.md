# Gesture Trigger

An application for capturing body and hand postures, training a model to recognize them, and triggering custom functions based on detected gestures. Designed for live art performances with real-time camera input.

## Features

- **Data Collection**: Easily capture your custom body and hand postures using a webcam
- **Model Training**: Train a machine learning model to recognize your unique gestures
- **Real-time Detection**: Efficient real-time gesture detection with MediaPipe and TensorFlow
- **Extensible Trigger System**: Trigger various actions when gestures are detected (sounds, visuals, MIDI, OSC, custom functions)

## Simple Project Structure

The streamlined project structure focuses on what you need most:

```
gesture-trigger/
├── config/              # Configuration files
│   ├── config.yaml      # Main configuration
│   └── triggers.yaml    # Trigger mappings
├── src/                 # Core source code modules
│   ├── data_collection/ # Data collection module
│   ├── model/           # Model training module
│   ├── detection/       # Gesture detection module
│   ├── triggers/        # Trigger system
│   └── visualization/   # Visual effects module
├── gui.py               # Graphical user interface (main entry point)
├── main.py              # Command-line interface (alternative entry point)
└── requirements.txt     # Dependencies
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/gesture-trigger.git
   cd gesture-trigger
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the setup script (one-time setup):
   ```
   python setup.py
   ```

## Quick Start

### Using the GUI (Recommended)

The graphical interface provides an easy way to collect data, train models, and detect gestures:

```
python gui.py
```

The GUI provides three main tabs:
1. **Collect Data**: Capture gesture samples using your webcam
2. **Train Model**: Train the recognition model with your collected data
3. **Run Detection**: Perform real-time gesture detection and trigger actions

### Using the Command Line

Alternatively, you can use the command-line interface:

```
# Collect gesture data
python main.py --mode collect --gesture your_gesture_name

# Train the model
python main.py --mode train

# Run gesture detection
python main.py --mode run
```

## Configuration

You can customize the application by editing the configuration files:

- `config/config.yaml`: General application settings, camera parameters, and model options
- `config/triggers.yaml`: Mapping of gestures to triggered actions

## Creating Custom Triggers

To map gestures to custom actions, edit the `config/triggers.yaml` file or use the "Edit Triggers" button in the GUI.

Available trigger types:
- **Sound**: Play audio files when gestures are detected
- **Visual**: Display visual effects on screen
- **MIDI**: Send MIDI signals to control music software
- **OSC**: Send Open Sound Control messages to compatible software and devices
- **Custom**: Call custom Python functions

## OSC Integration

You can use the OSC (Open Sound Control) feature to send messages to other applications and devices:

1. **Single Messages**: Send OSC messages with custom address patterns and arguments
2. **Message Bundles**: Send multiple OSC messages at once in a bundle
3. **Repeated Messages**: Configure messages to repeat at specific intervals

Example configuration in `triggers.yaml`:
```yaml
victory:
  type: osc
  params:
    ip: "127.0.0.1"     # OSC server IP address
    port: 9000          # OSC server port number
    address: "/gesture" # OSC address pattern
    args: ["victory"]   # Arguments to send (gesture name by default)
```

Compatible with software like Max/MSP, Pure Data, TouchDesigner, SuperCollider, and many others.

## Troubleshooting

If you encounter issues:

1. Ensure your webcam is properly connected and accessible
2. Make sure you have collected and trained with enough gesture samples (at least 30-50 per gesture)
3. Adjust the confidence threshold in the detection settings if gestures aren't being recognized properly

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

This means you are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial** — You may not use the material for commercial purposes.

For more information, see the [full license](https://creativecommons.org/licenses/by-nc/4.0/).
