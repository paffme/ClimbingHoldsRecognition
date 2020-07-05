import argparse
from json import JSONEncoder
import json
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

from mrcnn.config import Config

# Directory to save logs and trained model
LOGS_DIR = os.path.join(ROOT_DIR, "logs/default")


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class InferenceConfig(Config):
    """Configuration for training on the toy  datasets.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "climbing-hold-recognition"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 + 1  # background + climbing-hold + climbing-volume

    # Number of training steps per epoch
    # STEPS_PER_EPOCH = 100
    STEPS_PER_EPOCH = 100

    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = 0.8

    EPOCHS = 30


parser = argparse.ArgumentParser(
    description='Train Mask R-CNN to detect Climbing Holds.')
parser.add_argument("command",
                    metavar="<command>",
                    help="'visualize' or 'export' or 'visualize_and_export'")
parser.add_argument('--weights', required=True,
                    metavar="/path/to/weights.h5",
                    help="Path to weights .h5 file or 'coco'")
parser.add_argument('--image', required=True,
                    metavar="path or URL to image",
                    help='Image to apply the color splash effect on')
parser.add_argument('--file_save', required=False,
                    metavar="path or URL to image",
                    help='Image to apply the color splash effect on')
args = parser.parse_args()

if args.command == "export" or args.command == "visualize_and_export":
    assert args.file_save is not None

config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=LOGS_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(args.weights, by_name=True)

class_names = ['Background', 'ClimbingHold', 'ClimbingVolume']

image = skimage.io.imread(args.image)

# Run detection
results = model.detect([image], verbose=1)
r = results[0]

# Visualize results
if args.command == "visualize" or args.command == "visualize_and_export":
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])

# Export results
if args.command == "export" or args.command == "visualize_and_export":
    boundingBoxJson = json.dumps(r['rois'], cls=NumpyArrayEncoder)
    with open(args.file_save, 'w') as fs:
        fs.write(boundingBoxJson)
