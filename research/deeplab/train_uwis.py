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
    INIT_FOLDER = os.path.join(WORK_DIR, DATASET_DIR, PQR_FOLDER, "init_models")
    TRAIN_LOGDIR = os.path.join(WORK_DIR, DATASET_DIR, PQR_FOLDER, EXP_FOLDER, "train")
    DATASET = os.path.join(WORK_DIR, DATASET_DIR, PQR_FOLDER, "tfrecords_livingroom")

    # Ensure directories exist
    os.makedirs(os.path.join(WORK_DIR, DATASET_DIR, PQR_FOLDER, "exp"), exist_ok=True)
    os.makedirs(TRAIN_LOGDIR, exist_ok=True)

    # Training parameters
    NUM_ITERATIONS = 20000
    TRAIN_SCRIPT = os.path.join(WORK_DIR, "train.py")

    # Build the command
    cmd = [
        "python", TRAIN_SCRIPT,
        "--logtostderr",
        "--train_split=train",
        "--dataset=uwis_livingroom",
        "--model_variant=xception_65",
        "--num_clone=1",                       # changed 2 to 1 to fit in GPU memory
        "--atrous_rates=6",
        "--atrous_rates=12",
        "--atrous_rates=18",
        "--output_stride=16",
        "--decoder_output_stride=4",
        "--train_crop_size=321,321",         # changed 513,513 to 321,321 to fit in GPU memory
        "--optimizer=momentum",
        "--top_k_percent_pixels=0.01",
        "--hard_example_mining_step=2500",
        "--train_batch_size=2",              # changed 4 to 2 to fit in GPU memory
        "--log_steps=10",
        "--save_interval_secs=600",
        f"--training_number_of_steps={NUM_ITERATIONS}",
        f"--tf_initial_checkpoint={os.path.join(INIT_FOLDER, 'deeplabv3_pascal_train_aug/model.ckpt')}",
        "--initialize_last_layer=false",
        "--fine_tune_batch_norm=false",
        f"--train_logdir={TRAIN_LOGDIR}",
        f"--dataset_dir={DATASET}"
    ]

    # Run the training process
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
