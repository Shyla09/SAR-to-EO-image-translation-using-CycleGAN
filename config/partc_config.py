# Part C Configuration
# Enhanced SAR -> EO Translation

DATA_DIR = "data"

SAR_DIR = "data/processed"
EO_DIR = "data/processed"

CHECKPOINT_DIR = "checkpoints/partc"

GENERATED_IMAGES_DIR = "generated_images/partc"

METRICS_DIR = "metrics/partc"

INPUT_CHANNELS = 2

# Update this based on your Part C experiment
OUTPUT_CHANNELS = 13

IMAGE_SIZE = 256

BATCH_SIZE = 1

NUM_EPOCHS = 300

LEARNING_RATE = 2e-4

BETA1 = 0.5
BETA2 = 0.999

LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 5

SAVE_EVERY = 10

DEVICE = "cuda"
