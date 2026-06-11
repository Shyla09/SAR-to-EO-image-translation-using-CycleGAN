# Part A Configuration
# SAR (2 Bands) -> EO (13 Bands)

DATA_DIR = "data"

SAR_DIR = "data/processed"
EO_DIR = "data/processed"

CHECKPOINT_DIR = "checkpoints/parta"

GENERATED_IMAGES_DIR = "generated_images/parta"

METRICS_DIR = "metrics/parta"

INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 13

IMAGE_SIZE = 256

BATCH_SIZE = 1

NUM_EPOCHS = 200

LEARNING_RATE = 2e-4

BETA1 = 0.5
BETA2 = 0.999

LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 5

DEVICE = "cuda"
