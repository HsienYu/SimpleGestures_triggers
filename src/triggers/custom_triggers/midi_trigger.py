#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIDI Trigger Module

This module provides MIDI-based trigger functionality for controlling music software or hardware.
"""

import os
import time
import logging
from threading import Thread

logger = logging.getLogger('GestureTrigger.MidiTrigger')


class MidiTrigger:
    """Trigger for sending MIDI messages when gestures are detected."""

    def __init__(self, params):
        """Initialize the MIDI trigger with the given parameters."""
        self.params = params
        self.note = params.get('note', 60)  # Default to middle C
        self.velocity = params.get('velocity', 100)
        self.channel = params.get('channel', 1)
        self.duration = params.get('duration', 0.5)  # Note duration in seconds
        self.control_number = params.get(
            'control_number', None)  # For CC messages
        self.control_value = params.get(
            'control_value', None)  # For CC messages
        self.midi_port = params.get('midi_port', None)  # MIDI output port
        self._thread = None
        self._is_playing = False

        # Try to import MIDI libraries
        try:
            import rtmidi
            self.rtmidi = rtmidi
            self.midiout = rtmidi.MidiOut()
            self.use_rtmidi = True
            logger.info("Using rtmidi for MIDI output")

            # List available MIDI ports
            available_ports = self.midiout.get_ports()
            logger.info(f"Available MIDI ports: {available_ports}")

            # Open the specified port or the first available port
            if self.midi_port is not None and self.midi_port < len(available_ports):
                self.midiout.open_port(self.midi_port)
                logger.info(
                    f"Opened MIDI port: {available_ports[self.midi_port]}")
            elif available_ports:
                self.midiout.open_port(0)
                logger.info(f"Opened default MIDI port: {available_ports[0]}")
            else:
                logger.warning("No MIDI ports available, opening virtual port")
                self.midiout.open_virtual_port("GestureTrigger MIDI")

        except ImportError:
            self.use_rtmidi = False
            logger.warning("rtmidi not available")

            # Try to import mido as fallback
            try:
                import mido
                self.mido = mido
                self.use_mido = True
                logger.info("Using mido for MIDI output")

                # Open MIDI output
                try:
                    if self.midi_port:
                        self.midiout = mido.open_output(self.midi_port)
                    else:
                        # Get first available port or create virtual port
                        available_ports = mido.get_output_names()
                        if available_ports:
                            self.midiout = mido.open_output(available_ports[0])
                            logger.info(
                                f"Opened MIDI port: {available_ports[0]}")
                        else:
                            self.midiout = mido.open_output(
                                'GestureTrigger MIDI', virtual=True)
                            logger.info("Created virtual MIDI port")
                except Exception as e:
                    logger.error(f"Failed to open MIDI port: {e}")
                    self.use_mido = False
            except ImportError:
                self.use_mido = False
                logger.warning("mido not available")
                logger.error(
                    "No MIDI library available. MIDI trigger will not work.")

    def _send_note_rtmidi(self):
        """Send MIDI note using rtmidi."""
        try:
            # Note on message
            note_on = [0x90 + (self.channel - 1), self.note, self.velocity]
            self.midiout.send_message(note_on)
            logger.debug(
                f"Sent MIDI Note On: note={self.note}, velocity={self.velocity}, channel={self.channel}")

            # Wait for note duration
            time.sleep(self.duration)

            # Note off message
            note_off = [0x80 + (self.channel - 1), self.note, 0]
            self.midiout.send_message(note_off)
            logger.debug(
                f"Sent MIDI Note Off: note={self.note}, channel={self.channel}")

            self._is_playing = False
        except Exception as e:
            logger.error(f"Failed to send MIDI note with rtmidi: {e}")
            self._is_playing = False

    def _send_cc_rtmidi(self):
        """Send MIDI Control Change using rtmidi."""
        try:
            # Control Change message
            cc_message = [0xB0 + (self.channel - 1),
                          self.control_number, self.control_value]
            self.midiout.send_message(cc_message)
            logger.debug(
                f"Sent MIDI CC: control={self.control_number}, value={self.control_value}, channel={self.channel}")

            self._is_playing = False
        except Exception as e:
            logger.error(f"Failed to send MIDI CC with rtmidi: {e}")
            self._is_playing = False

    def _send_note_mido(self):
        """Send MIDI note using mido."""
        try:
            # Note on message
            self.midiout.send(self.mido.Message(
                'note_on', note=self.note, velocity=self.velocity, channel=self.channel-1))
            logger.debug(
                f"Sent MIDI Note On: note={self.note}, velocity={self.velocity}, channel={self.channel}")

            # Wait for note duration
            time.sleep(self.duration)

            # Note off message
            self.midiout.send(self.mido.Message(
                'note_off', note=self.note, velocity=0, channel=self.channel-1))
            logger.debug(
                f"Sent MIDI Note Off: note={self.note}, channel={self.channel}")

            self._is_playing = False
        except Exception as e:
            logger.error(f"Failed to send MIDI note with mido: {e}")
            self._is_playing = False

    def _send_cc_mido(self):
        """Send MIDI Control Change using mido."""
        try:
            # Control Change message
            self.midiout.send(self.mido.Message('control_change',
                                                control=self.control_number,
                                                value=self.control_value,
                                                channel=self.channel-1))
            logger.debug(
                f"Sent MIDI CC: control={self.control_number}, value={self.control_value}, channel={self.channel}")

            self._is_playing = False
        except Exception as e:
            logger.error(f"Failed to send MIDI CC with mido: {e}")
            self._is_playing = False

    def trigger(self, data=None):
        """
        Send MIDI message when triggered.

        Args:
            data: Optional additional data from the gesture detection.
                 Can contain 'velocity' to override the default velocity.
        """
        if not hasattr(self, 'use_rtmidi') and not hasattr(self, 'use_mido'):
            logger.error("No MIDI library available")
            return False

        # Allow dynamic velocity control based on gesture data
        if data and 'velocity' in data:
            self.velocity = int(data['velocity'])

        # Stop any currently playing note
        self.stop()

        # Start a new thread for sending MIDI
        if self.control_number is not None and self.control_value is not None:
            # Send Control Change
            if self.use_rtmidi:
                self._thread = Thread(target=self._send_cc_rtmidi)
            elif self.use_mido:
                self._thread = Thread(target=self._send_cc_mido)
        else:
            # Send Note On/Off
            if self.use_rtmidi:
                self._thread = Thread(target=self._send_note_rtmidi)
            elif self.use_mido:
                self._thread = Thread(target=self._send_note_mido)

        if self._thread:
            self._is_playing = True
            self._thread.daemon = True
            self._thread.start()
            return True

        return False

    def stop(self):
        """Stop any currently playing note."""
        if self._is_playing:
            # For MIDI, we don't forcibly stop the thread but instead
            # wait for it to complete its duration
            self._is_playing = False
            logger.debug("Stopping MIDI playback")

        return True
