#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sound Trigger Module

This module provides sound-based trigger functionality.
"""

import os
import logging
from threading import Thread

logger = logging.getLogger('GestureTrigger.SoundTrigger')


class SoundTrigger:
    """Trigger for playing sound effects when gestures are detected."""

    def __init__(self, params):
        """Initialize the sound trigger with the given parameters."""
        self.params = params
        self.sound_file = params.get('sound_file', None)
        self.volume = params.get('volume', 1.0)
        self.loop = params.get('loop', False)
        self._thread = None
        self._is_playing = False

        # Try to import pygame for sound, fall back to simpler methods if not available
        try:
            import pygame
            self.pygame = pygame
            self.pygame.mixer.init()
            self.use_pygame = True
            logger.info("Using pygame for sound playback")
        except ImportError:
            self.use_pygame = False
            logger.warning("pygame not available, using simpler sound methods")

            # Try to import simpleaudio as fallback
            try:
                import simpleaudio as sa
                self.sa = sa
                self.use_simpleaudio = True
                logger.info("Using simpleaudio for sound playback")
            except ImportError:
                self.use_simpleaudio = False
                logger.warning("simpleaudio not available")

                # Last resort: Use system commands
                if os.name == 'posix':  # macOS or Linux
                    self.use_system = True
                    logger.info("Using system commands for sound playback")
                else:
                    logger.error("No sound playback method available")
                    self.use_system = False

    def _play_sound_pygame(self):
        """Play sound using pygame."""
        try:
            sound = self.pygame.mixer.Sound(self.sound_file)
            sound.set_volume(self.volume)

            if self.loop:
                sound.play(-1)  # -1 means loop indefinitely
            else:
                sound.play()

            self._is_playing = True
            logger.debug(f"Playing sound: {self.sound_file}")
        except Exception as e:
            logger.error(f"Failed to play sound with pygame: {e}")
            self._is_playing = False

    def _play_sound_simpleaudio(self):
        """Play sound using simpleaudio."""
        try:
            import wave
            wave_obj = wave.open(self.sound_file, 'rb')
            play_obj = self.sa.play_buffer(
                wave_obj.readframes(wave_obj.getnframes()),
                wave_obj.getnchannels(),
                wave_obj.getsampwidth(),
                wave_obj.getframerate()
            )

            if not self.loop:
                play_obj.wait_done()
            self._is_playing = True
            logger.debug(f"Playing sound: {self.sound_file}")
        except Exception as e:
            logger.error(f"Failed to play sound with simpleaudio: {e}")
            self._is_playing = False

    def _play_sound_system(self):
        """Play sound using system commands."""
        try:
            import subprocess

            if os.name == 'posix':  # macOS or Linux
                if 'darwin' in os.sys.platform:  # macOS
                    cmd = ['afplay', self.sound_file]
                else:  # Linux
                    cmd = ['aplay', self.sound_file]

                subprocess.Popen(cmd)
                self._is_playing = True
                logger.debug(f"Playing sound: {self.sound_file}")
            else:
                logger.error("Unsupported platform for system sound playback")
                self._is_playing = False
        except Exception as e:
            logger.error(f"Failed to play sound with system command: {e}")
            self._is_playing = False

    def trigger(self, data=None):
        """
        Play a sound when triggered.

        Args:
            data: Optional additional data from the gesture detection.
        """
        if not self.sound_file:
            logger.warning("No sound file specified")
            return False

        if not os.path.exists(self.sound_file):
            logger.error(f"Sound file not found: {self.sound_file}")
            return False

        # Stop any currently playing sound
        self.stop()

        # Play sound in a separate thread
        if self.use_pygame:
            self._thread = Thread(target=self._play_sound_pygame)
        elif self.use_simpleaudio:
            self._thread = Thread(target=self._play_sound_simpleaudio)
        elif self.use_system:
            self._thread = Thread(target=self._play_sound_system)
        else:
            logger.error("No sound playback method available")
            return False

        self._thread.daemon = True
        self._thread.start()

        return True

    def stop(self):
        """Stop playing the current sound."""
        if self._is_playing:
            if self.use_pygame:
                self.pygame.mixer.stop()
            # For other methods, we rely on the thread to complete

            self._is_playing = False
            logger.debug("Stopped sound playback")

        return True
