import os
import tempfile
import shutil
from datetime import datetime
from tkinter import messagebox


def add_write_gesture_export_info(info_file, gestures):
    """Helper function to write gesture export information to a file."""
    with open(info_file, 'w') as f:
        f.write("Gesture Dataset Export\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Gestures:\n")

        for gesture, sample_count in gestures:
            f.write(f"- {gesture}: {sample_count} samples\n")

    return True


def process_import_dataset(zip_file, dataset_path, available_gestures):
    """Helper function to process dataset import."""
    # Ask for confirmation if dataset already has gestures
    if available_gestures:
        choice = messagebox.askyesnocancel(
            "Import Options",
            "How do you want to import the dataset?\n\n"
            "Yes: Merge with existing dataset\n"
            "No: Replace existing dataset\n"
            "Cancel: Abort import")

        if choice is None:  # Cancel
            return False

        if not choice:  # No - replace
            # Delete existing dataset
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
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract to temp directory
        shutil.unpack_archive(zip_file, temp_dir, 'zip')

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

    return True
