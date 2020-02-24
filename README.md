# Ocrd_anybaseocr_block_segmentation

This repository contains code infrastructure to implement training for block segemntation. This repository is created for OCR-D in reference to https://github.com/mjenckel/ocrd_anybaseocr. The scripts for training the Mask-RCNN are taken from https://github.com/matterport/Mask_RCNN.

This file explains all the necessary steps from preprocessing to the training of the dataset on Mask-RCNN model.

## Groundtruth Preprocessing

The current groundtruth is present in XML format which needs to be converted into JSON first.

The script xml_to_json.py takes path of an input directory where the groundtruth xml files are stored.

It creates JSON file for each xml file and also one JSON file at the end that contains the name of all the JSON files.

One JSON file allows us to load single json file at a time in order to avoid memory loads.

The name of the JSON files will be same with the name on corresponding images.

```sh
  $ python xml_to_json.py -input /path/to/groundtruth/ -output /desired_path/
```
## Creating Dataset

Once JSON files are ready, create two folders train and val.

Take a 80/20 split and put all the training images along with their JSON files in the train folder and the rest in val folder.

## Inspecting the Dataset

inspect_block_data.ipynb is an script taken from https://github.com/matterport/Mask_RCNN which helps in analysing whether the dataset is precisely created according to the requirement or not.

The scripts counts all the samples, helps you visualize the polygons on the image and so on.

To learn more about scirpt, you can visit https://github.com/matterport/Mask_RCNN.


## Training the Model

All the model related files are stored in mrcnn folder.

Apart from Config.py, all other files are same as in https://github.com/matterport/Mask_RCNN.

Parameters in Config.py is changed according to the requirements.

Once the dataset is verified, then training can be started. Training can be done in various ways :

```sh
# Train a new model starting from pre-trained COCO weights
$ python3 blocks.py train --dataset=/path/to/coco/ --model=coco

# Train a new model starting from ImageNet weights
$ python3 blocks.py train --dataset=/path/to/coco/ --model=imagenet

# Continue training a model that you had trained earlier
$ python3 blocks.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

# Continue training the last model you trained. This will find
# the last trained weights in the model directory.
$ python3 blocks.py train --dataset=/path/to/coco/ --model=last

```

## Inspecting the Model

When trainins is done, the model is stored in the logs folder. 

The model can be examined by using inspect_block_model.ipynb script. 

To learn more about this script, please refer to Matterport repository https://github.com/matterport/Mask_RCNN .
