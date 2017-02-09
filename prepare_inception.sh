#!/bin/bash

# This script is to be used post ssd training
# Once a SSD model is training, bounding boxes are detected for each image in the training and validation dataset
# An important note is that boxes are detected on images used in training, so overfitting is a risk
# Two directories are created, one where a bounding box is found, and one where no bounding boxes are found for the image


# The script runs from CAFFE_ROOT
cd $CAFFE_ROOT

# These parameters may need updating
caffe_model=models/fish_aug/snpsht/SSD_adam_augmentation_300x300/Fish_SSD_adam_augmentation_300x300_iter_120000.caffemodel
deploy=models/fish_aug/SSD_adam_augmentation_300x300/deploy.prototxt
training_dir=~/Desktop/Fish/train/train_ssd/images
validation_dir=~/Desktop/Fish/train/validation_ssd/images
out_dir=models/fish_aug/SSD_adam_augmentation_300x300
model_dir=models/fish_aug/SSD_adam_augmentation_300x300/


# Create text file of images to perform object detection
python tools/extra/create_textfile.py $training_dir $out_dir train
python tools/extra/create_textfile.py $validation_dir $out_dir tests

# Run command to output bounding box results
./build/examples/ssd/ssd_detect.bin $deploy $caffe_model $model_dir/train_images.txt > $model_dir/training_boxes.txt
./build/examples/ssd/ssd_detect.bin $deploy $caffe_model $model_dir/tests_images.txt > $model_dir/validation_boxes.txt

# Perform some reformatting of the textfile for plotting the detections
python tools/extra/create_textfile.py $training_dir $out_dir train
python tools/extra/create_textfile.py $validation_dir $out_dir tests
