#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom Light Control Trigger Module

This is an example custom trigger module that could control lighting.
"""

import logging

logger = logging.getLogger('GestureTrigger.CustomTriggers.LightControl')


def change_lights(gesture_name, color, intensity):
    """
    Change connected lights to specified color and intensity.

    This is a placeholder function that demonstrates how custom triggers
    can be implemented. In a real application, this would connect to
    lighting hardware or software.

    Args:
        gesture_name (str): The name of the detected gesture
        color (list): RGB color values [R, G, B]
        intensity (float): Light intensity from 0.0 to 1.0

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Light control triggered by gesture: {gesture_name}")
    logger.info(f"Setting lights to color {color} with intensity {intensity}")

    # Placeholder for actual hardware interaction
    # Example of how you could integrate with a library like phue for Philips Hue:
    #
    # try:
    #     from phue import Bridge
    #     bridge = Bridge('192.168.1.100')  # IP address of your Hue bridge
    #     bridge.connect()
    #
    #     # Convert RGB to Hue/Saturation/Brightness
    #     h, s, b = rgb_to_hsb(color[0], color[1], color[2])
    #
    #     # Set all lights
    #     for light in bridge.lights:
    #         light.brightness = int(intensity * 254)
    #         light.hue = h
    #         light.saturation = s
    # except Exception as e:
    #     logger.error(f"Failed to control lights: {e}")
    #     return False

    return True


def rgb_to_hsb(r, g, b):
    """
    Convert RGB values to Hue, Saturation, Brightness values.

    This is a utility function that would be used in a real implementation
    to convert RGB values to the format needed by lighting systems.

    Args:
        r (int): Red value (0-255)
        g (int): Green value (0-255)
        b (int): Blue value (0-255)

    Returns:
        tuple: (hue, saturation, brightness) values
    """
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0

    max_val = max(r, g, b)
    min_val = min(r, g, b)
    delta = max_val - min_val

    # Calculate hue
    if delta == 0:
        h = 0
    elif max_val == r:
        h = ((g - b) / delta) % 6
    elif max_val == g:
        h = (b - r) / delta + 2
    else:
        h = (r - g) / delta + 4

    h = round(h * 60) % 360

    # Calculate saturation
    if max_val == 0:
        s = 0
    else:
        s = delta / max_val

    # Calculate brightness
    v = max_val

    # Convert to values expected by lighting systems
    hue = int(h * 65535 / 360)  # Hue in 0-65535 range
    saturation = int(s * 254)   # Saturation in 0-254 range
    brightness = int(v * 254)   # Brightness in 0-254 range

    return (hue, saturation, brightness)
