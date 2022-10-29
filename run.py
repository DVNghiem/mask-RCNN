import os
import datetime
from mrcnn import utils
import tensorflow as tf
import mrcnn.model as modellib
from config import CustomConfig
from dataset import CustomDataset


config = CustomConfig()


def load_image_dataset(annotation_path, dataset_path, dataset_type):
    dataset_train = CustomDataset()
    dataset_train.load_custom(annotation_path, dataset_path, dataset_type)
    dataset_train.prepare()
    return dataset_train


annotations_path = "./label.json"
images_path = "./data"


dataset_train = load_image_dataset(annotations_path, images_path, "train")
dataset_val = load_image_dataset(annotations_path, images_path, "val")
class_number = dataset_train.count_classes()
print('Train: %d' % len(dataset_train.image_ids))
print('Validation: %d' % len(dataset_val.image_ids))
print("Classes: {}".format(class_number))
print(*["Classes: ", *dataset_train.class_names])

MODEL_DIR = os.path.join(".", "logs")
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
model.keras_model.metrics_tensors = []

# Tensorflow board
logdir = os.path.join(
    "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

init_with = "coco"
COCO_MODEL_PATH = "mask_rcnn_coco.h5"
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
                   "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

model.train(dataset_train, dataset_val,
            learning_rate=5e-5,
            epochs=50,
            layers='heads')
