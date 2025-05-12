#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Module

This module provides visual effects that can be triggered by gestures
for enhancing live performances.
"""

import os
import cv2
import time
import numpy as np
import logging
import threading
from collections import deque

logger = logging.getLogger('GestureTrigger.Visualization')


class VisualEffect:
    """Base class for visual effects."""

    def __init__(self, duration=2.0):
        """
        Initialize the visual effect.

        Args:
            duration (float): Duration of the effect in seconds
        """
        self.duration = duration
        self.start_time = 0
        self.is_active = False

    def trigger(self):
        """Trigger the effect."""
        self.start_time = time.time()
        self.is_active = True

    def update(self):
        """Update the effect state."""
        if self.is_active and time.time() - self.start_time > self.duration:
            self.is_active = False

    def render(self, frame):
        """
        Render the effect on the frame.

        Args:
            frame (numpy.ndarray): The frame to render the effect on

        Returns:
            numpy.ndarray: The modified frame
        """
        return frame


class ParticleBurst(VisualEffect):
    """Particle burst effect."""

    def __init__(self, duration=2.0, num_particles=100, color=(0, 0, 255), origin=None):
        """
        Initialize the particle burst effect.

        Args:
            duration (float): Duration of the effect in seconds
            num_particles (int): Number of particles
            color (tuple): RGB color of particles
            origin (tuple): Origin point of the burst (x, y), or None for center
        """
        super().__init__(duration)
        self.num_particles = num_particles
        self.color = color
        self.origin = origin
        self.particles = []

    def trigger(self, origin=None):
        """
        Trigger the particle burst.

        Args:
            origin (tuple): Override the origin point for this burst
        """
        super().trigger()

        # Use provided origin or the stored origin
        burst_origin = origin or self.origin

        # Initialize particles
        self.particles = []
        for _ in range(self.num_particles):
            # Random velocity and lifetime for each particle
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(1, 10)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            lifetime = np.random.uniform(0.5, self.duration)

            self.particles.append({
                'x': 0,
                'y': 0,
                'vx': vx,
                'vy': vy,
                'lifetime': lifetime,
                'birth_time': time.time()
            })

    def update(self):
        """Update particle positions and state."""
        super().update()

        current_time = time.time()
        for particle in self.particles:
            # Update position based on velocity
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']

            # Apply gravity effect
            particle['vy'] += 0.2

    def render(self, frame):
        """Render particles on the frame."""
        if not self.is_active:
            return frame

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Determine origin if not specified
        if self.origin is None:
            origin_x, origin_y = width // 2, height // 2
        else:
            origin_x, origin_y = self.origin

        # Update particles
        self.update()

        # Draw particles
        current_time = time.time()
        for particle in self.particles:
            # Skip dead particles
            if current_time - particle['birth_time'] > particle['lifetime']:
                continue

            # Calculate alpha based on remaining lifetime
            remaining = 1 - \
                (current_time - particle['birth_time']) / particle['lifetime']
            alpha = int(255 * remaining)

            # Calculate position
            x = int(origin_x + particle['x'])
            y = int(origin_y + particle['y'])

            # Skip if outside frame
            if x < 0 or x >= width or y < 0 or y >= height:
                continue

            # Draw particle with alpha blending
            overlay = frame.copy()
            cv2.circle(overlay, (x, y), 3, self.color, -1)
            cv2.addWeighted(overlay, alpha / 255, frame,
                            1 - alpha / 255, 0, frame)

        return frame


class RippleEffect(VisualEffect):
    """Ripple effect emanating from a point."""

    def __init__(self, duration=2.0, color=(0, 255, 0), origin=None, max_radius=300):
        """
        Initialize the ripple effect.

        Args:
            duration (float): Duration of the effect in seconds
            color (tuple): RGB color of the ripple
            origin (tuple): Origin point of the ripple (x, y), or None for center
            max_radius (int): Maximum radius of the ripple
        """
        super().__init__(duration)
        self.color = color
        self.origin = origin
        self.max_radius = max_radius
        self.start_time = 0

    def trigger(self, origin=None):
        """
        Trigger the ripple effect.

        Args:
            origin (tuple): Override the origin point for this ripple
        """
        super().trigger()
        if origin is not None:
            self.origin = origin

    def render(self, frame):
        """Render the ripple on the frame."""
        if not self.is_active:
            return frame

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Determine origin if not specified
        if self.origin is None:
            origin_x, origin_y = width // 2, height // 2
        else:
            origin_x, origin_y = self.origin

        # Calculate current radius based on elapsed time
        elapsed = time.time() - self.start_time
        progress = min(elapsed / self.duration, 1.0)
        radius = int(self.max_radius * progress)

        # Calculate alpha based on progress (fade out as it expands)
        alpha = int(255 * (1 - progress))

        # Draw the ripple with alpha blending
        overlay = frame.copy()
        cv2.circle(overlay, (origin_x, origin_y), radius, self.color, 2)
        cv2.addWeighted(overlay, alpha / 255, frame, 1 - alpha / 255, 0, frame)

        # Update state
        self.update()

        return frame


class TrailEffect(VisualEffect):
    """Motion trail effect following hand or body parts."""

    def __init__(self, duration=2.0, color=(255, 0, 0), max_points=20):
        """
        Initialize the trail effect.

        Args:
            duration (float): Duration of the effect in seconds
            color (tuple): RGB color of the trail
            max_points (int): Maximum number of points in the trail
        """
        super().__init__(duration)
        self.color = color
        self.max_points = max_points
        self.points = deque(maxlen=max_points)

    def add_point(self, x, y):
        """
        Add a point to the trail.

        Args:
            x (int): X coordinate
            y (int): Y coordinate
        """
        self.points.append((x, y, time.time()))
        self.is_active = True
        self.start_time = time.time()

    def update(self):
        """Update the trail state and remove old points."""
        super().update()

        # Remove points older than the duration
        current_time = time.time()
        while self.points and current_time - self.points[0][2] > self.duration:
            self.points.popleft()

        # Deactivate if no points remain
        if not self.points:
            self.is_active = False

    def render(self, frame):
        """Render the trail on the frame."""
        if not self.is_active or len(self.points) < 2:
            return frame

        # Update state
        self.update()

        # Draw the trail with fading opacity
        current_time = time.time()
        points_array = []
        for i, (x, y, t) in enumerate(self.points):
            # Calculate alpha based on age
            age = current_time - t
            alpha = 1.0 - (age / self.duration)

            # Add point to array for polylines
            points_array.append((int(x), int(y)))

            # Don't draw the first point
            if i == 0:
                continue

            # Draw line segment with alpha blending
            overlay = frame.copy()
            cv2.line(overlay, points_array[i-1],
                     points_array[i], self.color, 2)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame


class VisualizationManager:
    """Manages visual effects for the application."""

    def __init__(self):
        """Initialize the visualization manager."""
        self.effects = {}
        self.active_effects = []

        # Initialize some default effects
        self.effects['particle_burst'] = ParticleBurst()
        self.effects['ripple'] = RippleEffect()
        self.effects['trail'] = TrailEffect()

    def trigger_effect(self, effect_name, **params):
        """
        Trigger a visual effect.

        Args:
            effect_name (str): Name of the effect to trigger
            **params: Parameters to pass to the effect's trigger method

        Returns:
            bool: True if the effect was triggered, False otherwise
        """
        if effect_name not in self.effects:
            logger.error(f"Effect not found: {effect_name}")
            return False

        effect = self.effects[effect_name]
        effect.trigger(**params)

        # Add to active effects if not already there
        if effect not in self.active_effects:
            self.active_effects.append(effect)

        logger.info(f"Triggered effect: {effect_name}")
        return True

    def update_trail(self, effect_name, x, y):
        """
        Update a trail effect with a new point.

        Args:
            effect_name (str): Name of the trail effect
            x (int): X coordinate
            y (int): Y coordinate

        Returns:
            bool: True if the trail was updated, False otherwise
        """
        if effect_name not in self.effects:
            logger.error(f"Effect not found: {effect_name}")
            return False

        effect = self.effects[effect_name]
        if not isinstance(effect, TrailEffect):
            logger.error(f"Effect {effect_name} is not a TrailEffect")
            return False

        effect.add_point(x, y)

        # Add to active effects if not already there
        if effect not in self.active_effects:
            self.active_effects.append(effect)

        return True

    def add_custom_effect(self, name, effect):
        """
        Add a custom effect to the manager.

        Args:
            name (str): Name of the effect
            effect (VisualEffect): The effect instance
        """
        self.effects[name] = effect
        logger.info(f"Added custom effect: {name}")

    def process_frame(self, frame):
        """
        Process a frame with all active effects.

        Args:
            frame (numpy.ndarray): The frame to process

        Returns:
            numpy.ndarray: The processed frame
        """
        # Make a copy of the frame
        result = frame.copy()

        # Apply active effects
        for effect in list(self.active_effects):
            if effect.is_active:
                result = effect.render(result)
            else:
                # Remove inactive effects
                self.active_effects.remove(effect)

        return result
