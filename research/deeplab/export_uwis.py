import os
import subprocess

def main():
    # Set up working environment
    CURRENT_DIR = os.getcwd()
    WORK_DIR = os.path.join(CURRENT_DIR, "deeplab")
    DATASET_DIR = "datasets"

    # UW-IS dataset paths
    PQR_FOLDER = "uwis"
    EXP_FOLDER = "exp/train_on_trainval_set"
    TRAIN_LOGDIR = os.path.join(WORK_DIR, DATASET_DIR, PQR_FOLDER, EXP_FOLDER, "train")
    EXPORT_DIR = os.path.join(WORK_DIR, DATASET_DIR, PQR_FOLDER, EXP_FOLDER, "export")

    # Ensure export directory exists
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Export parameters
    NUM_ITERATIONS = 17548
    CKPT_PATH = os.path.join(TRAIN_LOGDIR, f"model.ckpt-{NUM_ITERATIONS}")
    EXPORT_PATH = os.path.join(EXPORT_DIR, "frozen_inference_graph.pb")

    EXPORT_SCRIPT = os.path.join(WORK_DIR, "export_model.py")

    # Build the command
    cmd = [
        "python", EXPORT_SCRIPT,
        "--logtostderr",
        f"--checkpoint_path={CKPT_PATH}",
        f"--export_path={EXPORT_PATH}",
        "--model_variant=xception_65",
        "--atrous_rates=6",
        "--atrous_rates=12",
        "--atrous_rates=18",
        "--output_stride=16",
        "--decoder_output_stride=4",
        "--num_classes=3",
        "--crop_size=513",
        "--crop_size=513",
        "--inference_scales=1.0"
    ]

    # Run the export process
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
