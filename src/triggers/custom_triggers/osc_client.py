#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom OSC (Open Sound Control) Trigger Module

This module provides OSC client functionality for sending messages to OSC servers
when gestures are detected.
"""

import logging
import time
import threading

logger = logging.getLogger('GestureTrigger.CustomTriggers.OSCClient')

# Try to import the python-osc library
try:
    from pythonosc import udp_client
    from pythonosc import osc_bundle_builder
    from pythonosc import osc_message_builder
    OSC_AVAILABLE = True
except ImportError:
    logger.warning(
        "python-osc library not found. Install with: pip install python-osc")
    OSC_AVAILABLE = False


def send_osc_message(gesture_name, ip="127.0.0.1", port=9000, address="/gesture",
                     args=None, repeat=1, interval=0.1):
    """
    Send an OSC message to a specified server.

    Args:
        gesture_name (str): The name of the detected gesture
        ip (str): IP address of the OSC server
        port (int): Port number of the OSC server
        address (str): OSC address pattern
        args (list, optional): List of arguments to send with the message
        repeat (int, optional): Number of times to repeat the message
        interval (float, optional): Time interval between repeated messages (seconds)

    Returns:
        bool: True if successful, False otherwise
    """
    if not OSC_AVAILABLE:
        logger.error(
            "Cannot send OSC message: python-osc library not installed")
        return False

    # Use default arguments if none provided
    if args is None:
        args = [gesture_name]
    elif not isinstance(args, list):
        args = [args]  # Convert non-list arg to list

    try:
        # Create OSC client
        client = udp_client.SimpleUDPClient(ip, port)

        # Format the OSC address if needed (ensure it starts with '/')
        if not address.startswith('/'):
            address = '/' + address

        logger.info(
            f"Sending OSC message to {ip}:{port}{address} with args {args}")

        # Send initial message
        client.send_message(address, args)

        # Send repeated messages if requested
        if repeat > 1:
            def send_repeated():
                for i in range(repeat - 1):  # Already sent first message
                    time.sleep(interval)
                    client.send_message(address, args)
                    logger.debug(f"Sent repeated OSC message {i+2}/{repeat}")

            # Start thread for repeated messages
            threading.Thread(target=send_repeated, daemon=True).start()

        return True

    except Exception as e:
        logger.error(f"Failed to send OSC message: {e}")
        return False


def send_osc_bundle(gesture_name, ip="127.0.0.1", port=9000, messages=None):
    """
    Send an OSC bundle (multiple messages) to a specified server.

    Args:
        gesture_name (str): The name of the detected gesture
        ip (str): IP address of the OSC server
        port (int): Port number of the OSC server
        messages (list): List of dicts with 'address' and 'args' keys

    Returns:
        bool: True if successful, False otherwise
    """
    if not OSC_AVAILABLE:
        logger.error(
            "Cannot send OSC bundle: python-osc library not installed")
        return False

    if not messages or not isinstance(messages, list):
        logger.error("No messages or invalid message format for OSC bundle")
        return False

    try:
        # Create OSC client
        client = udp_client.SimpleUDPClient(ip, port)

        # Create bundle builder
        bundle = osc_bundle_builder.OscBundleBuilder(
            osc_bundle_builder.IMMEDIATELY)

        # Add messages to bundle
        for msg in messages:
            address = msg.get('address', '/gesture')
            args = msg.get('args', [gesture_name])

            # Format the OSC address if needed
            if not address.startswith('/'):
                address = '/' + address

            # Create message builder
            msg_builder = osc_message_builder.OscMessageBuilder(
                address=address)

            # Add arguments
            if not isinstance(args, list):
                args = [args]

            for arg in args:
                msg_builder.add_arg(arg)

            # Add message to bundle
            bundle.add_content(msg_builder.build())

        # Build and send bundle
        client.send(bundle.build())

        logger.info(
            f"Sent OSC bundle to {ip}:{port} with {len(messages)} messages")
        return True

    except Exception as e:
        logger.error(f"Failed to send OSC bundle: {e}")
        return False
