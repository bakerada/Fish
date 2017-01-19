import os
import skimage.io as io
import argparse
import glob
import numpy as np

def create_directory(directory):
	if not os.path.isdir(directory):
		os.path.mkdir(directory)
def check_if_exist(directory):
	if not os.path.isdir(directory):
		raise ValueError('Directory does not exist')

def read_bbox(filename):
	data = np.genfromtxt(filename,dtype=float)
	bbox = []
	for row in data:
		bbox.append(row[1:])
	return bbox
def crop_image(img,bbox,output_dir):
	img_class = img.split('_')[0]
	class_directory = os.path.join(output_dir,img_class)
	create_directory(class_directory)
	image = io.imread(img)
	for box in bbox:
		xmin, ymin, xmax, ymax = np.maximum(box[0],0).astype(int),np.maximum(box[1],0).astype(int),
								 bbox[2].astype(int),bbox[3].astype(int)
		cropped = image[ymin:ymax,xmin:xmax]
		save_file = os.path.join(class_directory,img)
		io.imsave(save_file,cropped)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = "Crop Annotated Images for InceptionV3 Model.")
    parser.add_argument("directory",
            help = "A directory containing images.")
    parser.add_argument("annotations",
            help = "A directory containing annotations.")
    parser.add_argument("--output-dir", default="",
            help = "Directory to save cropped images")

    args = parser.parse_args()
    img_dir = parser.directory
    label_dir = parser.annotations
    output_dir = parser.output_dir

    check_if_exist(img_dir)
    check_if_exist(label_dir)
    create_directory(output_dir)


    images = glob.glob(img_dir + '*.jpg')
    annotations = glob.glob(label_dir + '*.txt')

    for img in images:
    	bbox = read_bbox(annotations[img])
    	crop_image(img,bbox,output_dir)
