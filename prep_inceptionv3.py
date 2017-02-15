import os
import skimage.io as io
import argparse
import glob
import numpy as np

def create_directory(directory):
	if not os.path.isdir(directory):
		os.mkdir(directory)
def check_if_exist(directory):
	if not os.path.isdir(directory):
		raise ValueError('Directory does not exist')

def read_bbox(filename):
	data = np.genfromtxt(filename,dtype=float)
	bbox = []
	if len(data.shape) == 1:
		return data[1:]
	for row in data:
		bbox.append(row[1:])
	return bbox
def crop_image(img,bbox,output_dir):
	filename = os.path.basename(img)
	img_class = filename.split('_')[0]
	class_directory = os.path.join(output_dir,img_class)
	create_directory(class_directory)
	image = io.imread(img)

	if isinstance(bbox,np.ndarray):
		filename = '{}_'.format(i) + filename
		xmin, ymin, xmax, ymax = np.maximum(bbox[0],0).astype(int),np.maximum(bbox[1],0).astype(int),bbox[2].astype(int),bbox[3].astype(int)
		cropped = image[ymin:ymax,xmin:xmax]
		save_file = os.path.join(class_directory,filename)
		io.imsave(save_file,cropped)

	else:	
		for box in bbox:
			xmin, ymin, xmax, ymax = np.maximum(box[0],0).astype(int),np.maximum(box[1],0).astype(int),box[2].astype(int),box[3].astype(int)
			cropped = image[ymin:ymax,xmin:xmax]
			save_file = os.path.join(class_directory,filename)
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
    img_dir = args.directory
    label_dir = args.annotations
    output_dir = args.output_dir

    check_if_exist(img_dir)
    check_if_exist(label_dir)
    create_directory(output_dir)


    images = glob.glob(img_dir + '/*.jpg')
    annotations = glob.glob(label_dir + '/*.txt')

    for img in images:
    	base = os.path.basename(img)
    	filename = base.split('.')[0]
    	txt = os.path.join(label_dir,'{}_ssd.txt'.format(filename))
    	bbox = read_bbox(txt)
    	crop_image(img,bbox,output_dir)
