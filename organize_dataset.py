

import os
import json
import shutil
from tqdm import tqdm

# --- Configuration ---
ANNOTATIONS_DIR = r"E:\New folder\vanitha\dataset\annotations\train"
SOURCE_IMAGE_DIR = r"E:\New folder\vanitha\dataset\hand_dataset\archive\train"
DESTINATION_DIR = r"E:\New folder\vanitha\dataset\train_structured"
# --- End Configuration ---

def organize_dataset():
    """
    Organizes a flat image directory into a structured one based on a directory of annotation files.
    """
    print(f"Scanning for annotation files in: {ANNOTATIONS_DIR}")
    if not os.path.exists(DESTINATION_DIR):
        print(f"Creating destination directory: {DESTINATION_DIR}")
        os.makedirs(DESTINATION_DIR)

    annotation_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.json')]
    print(f"Found {len(annotation_files)} annotation files.")

    moved_count = 0
    error_count = 0
    
    for json_file in tqdm(annotation_files, desc="Processing annotation files"):
        label = os.path.splitext(json_file)[0]
        annotation_path = os.path.join(ANNOTATIONS_DIR, json_file)

        with open(annotation_path, 'r') as f:
            # The JSON files contain a list of dictionaries, each with an 'image_id'
            image_entries = json.load(f)
        
        image_filenames = [entry['image_id'] for entry in image_entries]

        # Create the label subdirectory if it doesn't exist
        label_dir = os.path.join(DESTINATION_DIR, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        for image_filename in tqdm(image_filenames, desc=f"Moving '{label}' images", leave=False):
            source_path = os.path.join(SOURCE_IMAGE_DIR, image_filename)
            destination_path = os.path.join(label_dir, image_filename)

            if os.path.exists(source_path):
                try:
                    shutil.move(source_path, destination_path)
                    moved_count += 1
                except Exception as e:
                    print(f"Error moving {source_path}: {e}")
                    error_count += 1
            else:
                error_count += 1
            
    print("\n--- Organization Complete ---")
    print(f"Successfully moved {moved_count} images.")
    print(f"Could not find or move {error_count} images listed in the annotation files.")
    print("-----------------------------")


if __name__ == "__main__":
    organize_dataset()


