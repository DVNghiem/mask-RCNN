import mrcnn.model as modellib
from config import PredictConfig
import argparse
import matplotlib.pyplot as plt
from mrcnn import visualize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str,
                        default='test.jpg', help='path to test image')

    parser.add_argument('--model', type=str,
                        default='mask_rcnn_mica_0000.h5', help='path to model.h5, ex: mask_rcnn_mica_0050.h5')

    opts = parser.parse_args()

    config = PredictConfig()
    COCO_WEIGHTS_PATH = "."

    model = modellib.MaskRCNN(
        mode='inference', config=config, model_dir=COCO_WEIGHTS_PATH)
    model.load_weights(
        opts.model, by_name=True)

    image = plt.imread(opts.image)
    results = model.detect([image], verbose=1)
    r = results[0]
    visualize.display_instances(
        image, r['rois'], r['masks'], r['class_ids'], 'sau rang', r['scores'])
