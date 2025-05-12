#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trigger Manager Module

This module handles executing actions when gestures are detected.
It loads trigger configurations and provides an extensible plugin system.
"""

import os
import yaml
import time
import logging
import importlib
import threading
from collections import deque

logger = logging.getLogger('GestureTrigger.TriggerManager')


class TriggerManager:
    """Manages and executes trigger actions for detected gestures."""

    def __init__(self, config, trigger_log_manager=None):
        """Initialize the trigger manager with the given configuration."""
        self.config = config
        self.trigger_config_path = config['triggers']['trigger_config_path']

        # Trigger log manager for GUI updates
        self.trigger_log_manager = trigger_log_manager

        # Load trigger configurations
        self.triggers = self._load_trigger_config()

        # Initialize built-in trigger handlers
        self.handlers = {
            'sound': self._handle_sound_trigger,
            'visual': self._handle_visual_trigger,
            'midi': self._handle_midi_trigger,
            'custom': self._handle_custom_trigger,
            'osc': self._handle_osc_trigger,
        }

        # Recent triggers to avoid repeating too quickly
        self.recent_triggers = deque(maxlen=10)
        self.cooldown_times = {}

        # Loaded custom modules
        self.custom_modules = {}

    def _load_trigger_config(self):
        """Load trigger configurations from the YAML file."""
        if not os.path.exists(self.trigger_config_path):
            logger.error(
                f"Trigger config file not found: {self.trigger_config_path}")
            return {}

        try:
            with open(self.trigger_config_path, 'r') as f:
                config = yaml.safe_load(f)

            logger.info(
                f"Loaded {len(config.get('triggers', {}))} trigger configurations")
            return config.get('triggers', {})

        except Exception as e:
            logger.error(f"Failed to load trigger config: {e}")
            return {}

    def _handle_sound_trigger(self, gesture_name, params):
        """Handle a sound type trigger."""
        try:
            import pygame

            # Initialize pygame mixer if not already done
            if not hasattr(self, '_pygame_initialized'):
                pygame.mixer.init()
                self._pygame_initialized = True

            sound_file = params.get('sound_file')
            if not sound_file or not os.path.exists(sound_file):
                logger.error(f"Sound file not found: {sound_file}")
                return False

            # Play the sound
            sound = pygame.mixer.Sound(sound_file)
            sound.play()
            logger.info(f"Played sound: {sound_file}")
            return True

        except ImportError:
            logger.error(
                "pygame library not found. Install with: pip install pygame")
            return False
        except Exception as e:
            logger.error(f"Failed to play sound: {e}")
            return False

    def _handle_visual_trigger(self, gesture_name, params):
        """Handle a visual effect type trigger."""
        # This is a placeholder for visual effects
        # In a real implementation, this might update a shared state
        # that a visualization component would read

        effect = params.get('effect', 'default')
        color = params.get('color', [255, 255, 255])

        logger.info(f"Visual effect triggered: {effect}, color: {color}")

        # In a real implementation, you might do something like:
        # self.visual_effects.add_effect(effect, color)

        return True

    def _handle_midi_trigger(self, gesture_name, params):
        """Handle a MIDI type trigger."""
        try:
            import rtmidi

            # Initialize MIDI out if not already done
            if not hasattr(self, '_midi_out'):
                self._midi_out = rtmidi.MidiOut()
                available_ports = self._midi_out.get_ports()

                if available_ports:
                    self._midi_out.open_port(0)
                    logger.info(f"Opened MIDI port: {available_ports[0]}")
                else:
                    self._midi_out.open_virtual_port("GestureTrigger MIDI")
                    logger.info("Opened virtual MIDI port")

            note = params.get('note', 60)
            velocity = params.get('velocity', 64)
            channel = params.get('channel', 0)
            duration = params.get('duration', 0.1)

            # Create MIDI message
            note_on = [0x90 + channel, note, velocity]
            note_off = [0x80 + channel, note, 0]

            # Send note on message
            self._midi_out.send_message(note_on)
            logger.info(
                f"MIDI Note On: note={note}, velocity={velocity}, channel={channel}")

            # Schedule note off message
            def send_note_off():
                time.sleep(duration)
                self._midi_out.send_message(note_off)
                logger.info(f"MIDI Note Off: note={note}, channel={channel}")

            threading.Thread(target=send_note_off, daemon=True).start()
            return True

        except ImportError:
            logger.error(
                "rtmidi library not found. Install with: pip install python-rtmidi")
            return False
        except Exception as e:
            logger.error(f"Failed to send MIDI message: {e}")
            return False

    def _handle_osc_trigger(self, gesture_name, params):
        """Handle an OSC (Open Sound Control) type trigger."""
        try:
            from src.triggers.custom_triggers.osc_client import send_osc_message, send_osc_bundle

            # Get OSC parameters
            ip = params.get('ip', '127.0.0.1')
            port = params.get('port', 9000)
            address = params.get('address', '/gesture')
            args = params.get('args', [gesture_name])

            # Check if we should send a bundle or a single message
            if 'messages' in params:
                result = send_osc_bundle(
                    gesture_name=gesture_name,
                    ip=ip,
                    port=port,
                    messages=params['messages']
                )
            else:
                # Optional parameters
                repeat = params.get('repeat', 1)
                interval = params.get('interval', 0.1)

                result = send_osc_message(
                    gesture_name=gesture_name,
                    ip=ip,
                    port=port,
                    address=address,
                    args=args,
                    repeat=repeat,
                    interval=interval
                )

            return result

        except ImportError:
            logger.error(
                "python-osc library not found. Install with: pip install python-osc")
            return False
        except Exception as e:
            logger.error(f"Failed to send OSC message: {e}")
            return False

    def _handle_custom_trigger(self, gesture_name, params):
        """Handle a custom trigger type by importing and calling a custom module."""
        module_name = params.get('module')
        function_name = params.get('function')

        if not module_name or not function_name:
            logger.error(
                f"Missing module or function name for custom trigger: {gesture_name}")
            return False

        try:
            # Import the module if not already loaded
            if module_name not in self.custom_modules:
                self.custom_modules[module_name] = importlib.import_module(
                    module_name)

            module = self.custom_modules[module_name]

            # Get the function
            if not hasattr(module, function_name):
                logger.error(
                    f"Function {function_name} not found in module {module_name}")
                return False

            function = getattr(module, function_name)

            # Call the function with parameters
            function_params = params.get('params', {})
            result = function(gesture_name=gesture_name, **function_params)

            logger.info(
                f"Custom function executed: {module_name}.{function_name}()")
            return result

        except ImportError:
            logger.error(f"Failed to import module: {module_name}")
            return False
        except Exception as e:
            logger.error(f"Failed to execute custom trigger function: {e}")
            return False

    def execute_trigger(self, gesture_name, confidence=None):
        """Execute the trigger action for the detected gesture."""
        # Check if the gesture has a trigger configured
        if gesture_name not in self.triggers:
            return False

        # Check cooldown to prevent too frequent triggering
        current_time = time.time()
        last_time = self.cooldown_times.get(gesture_name, 0)
        cooldown = self.triggers[gesture_name].get(
            'cooldown', 1.0)  # Default 1 second cooldown

        if current_time - last_time < cooldown:
            # Still in cooldown period
            return False

        # Update cooldown time
        self.cooldown_times[gesture_name] = current_time

        # Get trigger configuration
        trigger_config = self.triggers[gesture_name]
        trigger_type = trigger_config.get('type')

        if not trigger_type:
            logger.error(
                f"No trigger type specified for gesture: {gesture_name}")
            return False

        # Get trigger details for logging
        trigger_details = ""
        if trigger_type == 'sound':
            sound_file = trigger_config.get('params', {}).get('sound_file', '')
            if sound_file:
                trigger_details = os.path.basename(sound_file)
        elif trigger_type == 'midi':
            note = trigger_config.get('params', {}).get('note', '')
            velocity = trigger_config.get('params', {}).get('velocity', '')
            if note:
                trigger_details = f"Note: {note}, Vel: {velocity}"
        elif trigger_type == 'custom':
            module = trigger_config.get('params', {}).get('module', '')
            function = trigger_config.get('params', {}).get('function', '')
            if module and function:
                trigger_details = f"{module}.{function}"

        # Find the right handler for the trigger type
        handler = self.handlers.get(trigger_type)

        if not handler:
            logger.error(f"No handler found for trigger type: {trigger_type}")
            return False

        # Execute the handler
        logger.info(
            f"Executing trigger for gesture: {gesture_name}, type: {trigger_type}")
        params = trigger_config.get('params', {})
        result = handler(gesture_name, params)

        # Log the trigger event if we have a log manager
        if result and self.trigger_log_manager:
            self.trigger_log_manager.log_trigger(
                detected_gesture=gesture_name,
                confidence=confidence if confidence is not None else 1.0,
                action_type=trigger_type,
                details=trigger_details
            )

        return result

    def register_custom_handler(self, trigger_type, handler_func):
        """Register a custom trigger handler."""
        if trigger_type in self.handlers:
            logger.warning(
                f"Overriding existing handler for trigger type: {trigger_type}")

        self.handlers[trigger_type] = handler_func
        logger.info(
            f"Registered custom handler for trigger type: {trigger_type}")

    def reload_triggers(self):
        """Reload trigger configurations from the config file."""
        self.triggers = self._load_trigger_config()
        logger.info("Reloaded trigger configurations")

    def cleanup(self):
        """Clean up resources used by the trigger manager."""
        # Close MIDI ports if opened
        if hasattr(self, '_midi_out'):
            self._midi_out.close_port()
            del self._midi_out
            logger.info("Closed MIDI port")

        # Close pygame mixer if initialized
        if hasattr(self, '_pygame_initialized'):
            import pygame
            pygame.mixer.quit()
            del self._pygame_initialized
            logger.info("Closed pygame mixer")
