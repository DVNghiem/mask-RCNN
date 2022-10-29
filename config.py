from mrcnn.config import Config


class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    BACKBONE = 'resnet50'
    NAME = "mica"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + 1 other class

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 10
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
