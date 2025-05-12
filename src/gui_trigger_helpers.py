import logging
import threading
import time
import tkinter as tk

logger = logging.getLogger('GestureTrigger.TriggerLog')


class TriggerLogManager:
    """Manages logging of triggered actions for display in the GUI."""

    def __init__(self, detection_log=None, gesture_list=None):
        """Initialize with UI components for updating."""
        self.detection_log = detection_log
        self.gesture_list = gesture_list
        self.last_triggers = {}  # Store last trigger time by gesture
        self.lock = threading.Lock()  # Thread safety

    def log_detected_gesture(self, detected_gesture, confidence):
        """Log a detected gesture to the UI components."""
        with self.lock:
            current_time = time.strftime("%H:%M:%S")
            confidence_pct = f"{confidence:.1%}"

            # Format the message for detection log
            log_message = f"[{current_time}] Detected: {detected_gesture} ({confidence_pct})\n"

            # Update detection log if available
            if self.detection_log:
                try:
                    self.detection_log.config(state='normal')
                    self.detection_log.insert('end', log_message, 'detected')
                    self.detection_log.see('end')  # Scroll to the end
                    self.detection_log.config(state='disabled')
                except Exception as e:
                    logger.error(f"Error updating detection log: {e}")

            # Update gesture list if available
            if self.gesture_list:
                try:
                    # Insert new row with all three values
                    self.gesture_list.insert(
                        '', 'end', values=(detected_gesture, confidence_pct, current_time))
                except Exception as e:
                    logger.error(f"Error updating gesture list: {e}")

            return current_time  # Return the timestamp for other uses

    def log_trigger(self, detected_gesture, confidence, action_type, details=""):
        """Log a trigger event to the UI components."""
        with self.lock:
            current_time = time.strftime("%H:%M:%S")
            confidence_pct = f"{confidence:.1%}" if confidence is not None else "N/A"

            # Format the message
            log_message = f"[{current_time}] TRIGGERED: {detected_gesture} - {action_type}"
            if details:
                log_message += f" - {details}"
            log_message += "\n"

            # Log to console for debugging
            logger.info(
                f"Trigger executed: {detected_gesture} - {action_type}")

            # Update detection log if available
            if self.detection_log:
                try:
                    self.detection_log.config(state='normal')
                    self.detection_log.insert('end', log_message, 'triggered')
                    self.detection_log.see('end')  # Scroll to the end
                    self.detection_log.config(state='disabled')
                except Exception as e:
                    logger.error(f"Error updating detection log: {e}")

            # Highlight the gesture in the gesture list if available
            if self.gesture_list:
                try:
                    # Find and highlight the matching gesture in the list
                    for item in self.gesture_list.get_children():
                        item_values = self.gesture_list.item(item, 'values')
                        if item_values and item_values[0] == detected_gesture:
                            # Add visual indication that this gesture was triggered
                            self.gesture_list.item(item, tags=('triggered',))
                            # Ensure it's visible
                            self.gesture_list.see(item)
                            break
                except Exception as e:
                    logger.error(f"Error highlighting triggered gesture: {e}")

            # Store last trigger time
            self.last_triggers[detected_gesture] = {
                'time': current_time,
                'confidence': confidence,
                'action': action_type,
                'details': details
            }

    def get_last_trigger_info(self, detected_gesture):
        """Get information about the last trigger for a gesture."""
        with self.lock:
            return self.last_triggers.get(detected_gesture, None)

    def clear_logs(self):
        """Clear the detection log and gesture list."""
        with self.lock:
            # Clear detection log
            if self.detection_log:
                try:
                    self.detection_log.config(state='normal')
                    self.detection_log.delete(1.0, 'end')
                    self.detection_log.insert(
                        'end', "Starting gesture detection...\n", 'info')
                    self.detection_log.config(state='disabled')
                except Exception as e:
                    logger.error(f"Error clearing detection log: {e}")

            # Clear gesture list
            if self.gesture_list:
                try:
                    for item in self.gesture_list.get_children():
                        self.gesture_list.delete(item)
                except Exception as e:
                    logger.error(f"Error clearing gesture list: {e}")

            # Clear last triggers
            self.last_triggers.clear()
            return None
