"""
Mask R-CNN
Train on the toy blocks dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 blocks.py train --dataset=/path/to/blocks/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 blocks.py train --dataset=/path/to/blocks/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 blocks.py train --dataset=/path/to/blocks/dataset --weights=imagenet

    # Apply color splash to an image
    python3 blocks.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 blocks.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
# COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "samples/blocks/logs")

############################################################
#  Configurations
############################################################


class BlockConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "block"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
#     NUM_CLASSES = 1 + 14  # Background + blocks
    NUM_CLASSES = 1 + 14  # only for running on old dataset
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class BlockDataset(utils.Dataset):
    def load_block(self, dataset_dir, subset):
        """Load a subset of the blocks dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add all the classes.
        #self.add_class("blocks", 1, "blocks")
        #self.add_class("block", 0, "BG")
        self.add_class("block", 7, "header")
        self.add_class("block", 1, "page-number")
        self.add_class("block", 2, "paragraph")
        self.add_class("block", 8, "marginalia")
        self.add_class("block", 6, "signature-mark")
        self.add_class("block", 3, "catch-word")
        self.add_class("block", 4, "heading")
        self.add_class("block", 5, "drop-capital")
        self.add_class("block", 13, "footer")
        self.add_class("block", 9, "footnote")
        self.add_class("block", 10, "footnote-continued")
        self.add_class("block", 11, "caption")
        self.add_class("block", 12, "endnote")
        self.add_class("block", 14, "TOC-entry")
        # Train or validation dataset?
        polygons=list()
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "base_names.json")))
        
        # The VIA tool saves images in the JSON even if they don't have any
        # Add images
        # Get the x, y coordinaets of points of the polygons that make up
        # the outline of each object instance. These are stores in the
        # shape_attributes (see json format above)
        # The if condition is needed to support VIA versions 1.x and 2.x.
    
        #iterate over single JSON file and make a dictionary for each JSON file along with the corresponding image
        for a in annotations:
            image_path = os.path.join(dataset_dir, a)
            image_path+=".tif"
            image = skimage.io.imread(image_path, plugin='pil')
            json_path = os.path.join(dataset_dir, a + ".json") 
            height, width = image.shape[:2]
            
        #print((block_classes))

            self.add_image(
                "block",
                image_id=a,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                json_path=json_path)
            


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a blocks dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "block":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        polygons = list()

        polygons_json = json.load(open(image_info["json_path"]))
        height, width = (image_info["height"], image_info["width"])
#         block_classes = list()
        class_dict = {"BG" :0,
                      "header" : 7,
                      "page-number": 1,
                      "paragraph" : 2,
                      "marginalia" :8,
                      "signature-mark" :6,
                      "catch-word": 3,
                      "heading" : 4,
                      "drop-capital" :5,
                      "footer" : 13,
                      "footnote": 9,
                      "footnote-continued" :10,
                      "caption" : 11,
                      "endnote" : 12,
                      "TOC-entry":14
                     }
        
        for x in polygons_json:

            #check if any point lies outside the image.
            for i in range(len(x['all_x_values'])):
                if x['all_x_values'][i] >= width:
                    x['all_x_values'][i] = width - 1
            for i in range(len(x['all_y_values'])):
                if x['all_y_values'][i] >= height:
#                     print(x['all_y_values'][i])
                    x['all_y_values'][i] = height -1

            polygons.append({
                'name': x['block_class'],
                'class_id':class_dict[x['block_class']],
                'all_x_values': [r for r in x['all_x_values']],
                'all_y_values': [r for r in x['all_y_values']]
            })

        #defining list to get class ids accordingly
        class_ids=list()
        
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(polygons)],
                        dtype=np.uint8)

        for i in range(len(polygons)):
            #looping over all polygons to create masks
            rr, cc = skimage.draw.polygon(polygons[i]['all_y_values'], polygons[i]['all_x_values'])
            for l in range(len(cc)):
                if cc[l] >= info["width"]:
                    cc[l] = info["width"] -1
            mask[rr, cc, i] = polygons[i]["class_id"]
            #print(info["polygons"][i]['all_x_values'],info["polygons"][i]['all_y_values'])
            class_ids.append(polygons[i]["class_id"])
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        
        #class_ids = np.array(class_ids, dtype=np.int32)
        return mask.astype(np.bool), np.array(class_ids, dtype=np.int32) 

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "block":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = BlockDataset()
    dataset_train.load_block(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BlockDataset()
    dataset_val.load_block(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Totally depends on the dataset.
    # IF you are using pre-trained weights. Then,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


#IF someone wants to test the model on any specific image then function would be helpful. 
def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        rois = r['rois']
        class_ids = r['class_ids']
        class_names = ['BG','page-number', 'paragraph', 'catch-word', 'heading', 'drop-capital', 'signature-mark','header',
                       'marginalia', 'footnote', 'footnote-continued', 'caption', 'endnote', 'footer','TOC-entry']
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    for roi in rois:
        print("ROI : ",roi)
    for class_id in class_ids:
        print("Class name : ",class_names[class_id])
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect blockss.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'infer'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/blocks/dataset/",
                        help='Directory of the blocks dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BlockConfig()
    else:
        class InferenceConfig(BlockConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
#         model.load_weights(weights_path, by_name=True, exclude=[
#                     "mrcnn_class_logits", "mrcnn_bbox_fc",
#                     "mrcnn_bbox", "mrcnn_mask"])

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "infer":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
