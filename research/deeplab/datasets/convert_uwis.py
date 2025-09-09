import os
import subprocess

# ---------- Folder definitions ----------
CURRENT_DIR = os.getcwd()
WORK_DIR = os.path.join(CURRENT_DIR, "uwis")
PQR_ROOT = os.path.join(CURRENT_DIR, "data")
SEG_FOLDER = os.path.join(PQR_ROOT, "foreground_livingroom")
SEMANTIC_SEG_FOLDER = os.path.join(PQR_ROOT, "foregroundraw_livingroom")

OUTPUT_DIR = os.path.join(WORK_DIR, "tfrecords_livingroom")
IMAGE_FOLDER = os.path.join(PQR_ROOT, "JPEGImages_livingroom")
LIST_FOLDER = os.path.join(PQR_ROOT, "ImageSets_livingroom")

# ---------- Create output folder ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Run the build_ara_scene_data.py script ----------
print("Converting scene dataset...")
subprocess.run([
    "python",
    "./build_ara_scene_data.py",
    f"--image_folder={IMAGE_FOLDER}",
    f"--semantic_segmentation_folder={SEMANTIC_SEG_FOLDER}",
    f"--list_folder={LIST_FOLDER}",
    "--image_format=jpg",
    f"--output_dir={OUTPUT_DIR}"
])
