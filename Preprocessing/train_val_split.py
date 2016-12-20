import json
import csv
import pandas as pd
import os
import glob
import shutil
import numpy as np
import cv2

cwd = os.getcwd()
print cwd

# This should be run in directory that houses training data
# DetectNet Directories
training_directory = os.path.join(cwd + '/train')
training_images = os.path.join(training_directory + '/images/')
training_labels = os.path.join(training_directory + '/labels/')
validation_directory = os.path.join(cwd + '/validation/')
validation_images = os.path.join(validation_directory + '/images/')
validation_labels = os.path.join(validation_directory + '/labels/')


if not os.path.isdir(training_directory):
    os.makedirs(training_directory)
if not os.path.isdir(validation_directory):
    os.makedirs(validation_directory) 
    

if not os.path.isdir(training_images):
    os.makedirs(training_images)

if not os.path.isdir(training_labels):
    os.makedirs(training_labels)

if not os.path.isdir(validation_images):
    os.makedirs(validation_images)

if not os.path.isdir(validation_labels):
    os.makedirs(validation_labels)

# SSD Directories
training_directory_ssd = os.path.join(cwd + '/train_ssd')
training_images_ssd = os.path.join(training_directory_ssd + '/images/')
training_labels_ssd = os.path.join(training_directory_ssd + '/labels/')
validation_directory_ssd = os.path.join(cwd + '/validation_ssd')
validation_images_ssd = os.path.join(validation_directory_ssd + '/images/')
validation_labels_ssd = os.path.join(validation_directory_ssd + '/labels/')


if not os.path.isdir(training_directory_ssd):
    os.makedirs(training_directory_ssd)
if not os.path.isdir(validation_directory_ssd):
    os.makedirs(validation_directory_ssd) 
    

if not os.path.isdir(training_images_ssd):
    os.makedirs(training_images_ssd)

if not os.path.isdir(training_labels_ssd):
    os.makedirs(training_labels_ssd)

if not os.path.isdir(validation_images_ssd):
    os.makedirs(validation_images_ssd)

if not os.path.isdir(validation_labels_ssd):
    os.makedirs(validation_labels_ssd)


jsons = glob.glob('*.json')
models = ['detectnet','ssd']
print "Parsing data for %s" % models


print ("Provided Labels: \n{}").format(jsons)

for model in models:
    print model
    for j in jsons:
        print j
        with open(j) as json_data:
            d = json.load(json_data)
            parsed = "{}.csv".format(j)
            f = csv.writer(open(parsed, "wb+"))
            for x in d:
                filename = x['filename']
                species,label = x['filename'].split('/')
                for a in x['annotations']:
                    f.writerow([
                            filename,
                            species,
                            label,
                            a["height"],
                            a["width"],
                            a["x"],
                            a["y"]
                        ])

        data = pd.read_csv(parsed,
                           names=['filename','species','label','height','width','x_min','y_min'],
                           header=None)
        data['x_max'] = data['x_min'] + data['width']
        data['y_max'] = data['y_min'] + data['height']
        data['object'] = 'fish'
        data = data.drop(['height','width'],axis=1)
        
        
        # Create train / validation structure
        n = len(data)
        cut = round(n * 0.8)
        
        for i in range(n-1):
            row = data.iloc[i].values.tolist()
            #print row
            img = cv2.imread(row[0])
            height, width, channels = img.shape
            row[3] = np.maximum(row[3],0)
            row[4] = np.maximum(row[4],0)
            row[5] = np.minimum(row[5],row[3] + width)
            row[6] = np.maximum(row[6],row[4] + height)
            # Traning set
            if i <= cut:
                if model == 'detectnet':
                    label_file = os.path.join(training_labels + row[1] +'_' + row[2].split('.')[0] + '.txt')
                    image_file = os.path.join(training_images + row[1] +'_' + row[2])
                    full = ['fish',0.0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                    full[4:8] = row[3:7]
                    row = full
                    
                    mode = 'a+' if os.path.isfile(label_file) else 'w+'
                    thefile = open(label_file, mode)
                    for i in row:
                      thefile.write("%s " % i)
                    thefile.write("\n")
                    thefile.close()
                    if not os.path.isfile(image_file):
                        shutil.copy2(data['filename'][i],image_file)

                if model == 'ssd':
                    # SSD requires key value pairs in a file
                    label_file = os.path.join(training_labels_ssd + row[1] +'_' + row[2].split('.')[0] + '_ssd.txt')
                    image_file = os.path.join(training_images_ssd + row[1] +'_' + row[2])
                    
                    if not os.path.isfile(image_file):
                        shutil.copy2(data['filename'][i],image_file)
                    
                    row[7]=1
                    indices = [7,3,4,5,6]
                    mode = 'a+' if os.path.isfile(label_file) else 'w+'
                    thefile = open(label_file, mode)
                    for i in indices:
                      thefile.write("%s " % row[i])
                    thefile.write("\n")
                    thefile.close()
                    
                    ssd_labels = 'train_ssd/labels/'
                    ssd_labels_2 = os.path.join(ssd_labels + row[1] +'_' + row[2].split('.')[0] + '_ssd.txt')
                    ssd_images = 'train_ssd/images/'
                    ssd_images_2 = os.path.join(ssd_images + row[1] +'_' + row[2])
                    mode = 'a+' if os.path.isfile('trainval.txt') else 'w+'
                    trainval = open('trainval.txt', mode)
                    trainval.write("{} {}".format(ssd_images_2,ssd_labels_2))
                    trainval.write("\n")
                    trainval.close()
                
            #validation set    
            else:
                label_file = os.path.join(validation_labels + row[1] +'_' + row[2].split('.')[0] + '.txt')
                image_file = os.path.join(validation_images + row[1] +'_' + row[2])
                if model == 'detectnet':
                    full = ['fish',0.0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                    full[4:8] = row[3:7]
                    row = full   
            
                if not os.path.isfile(image_file):
                    shutil.copy2(data['filename'][i],image_file)
                
                mode = 'a+' if os.path.isfile(label_file) else 'w+'
                thefile = open(label_file, mode)
                for i in row:
                  thefile.write("%s " % i)
                thefile.write("\n")
                thefile.close()

                    
                if model == 'ssd':
                    # SSD requires key value pairs in a file
                    label_file = os.path.join(validation_labels_ssd + row[1] +'_' + row[2].split('.')[0] + '_ssd.txt')
                    image_file = os.path.join(validation_images_ssd + row[1] +'_' + row[2])
                    indices = [7,3,4,5,6]
                    row[7]=1
                    mode = 'a+' if os.path.isfile(label_file) else 'w+'
                    thefile = open(label_file, mode)
                    for i in indices:
                      thefile.write("%s " % row[i])
                    thefile.write("\n")
                    thefile.close()
                    if not os.path.isfile(image_file):
                        shutil.copy2(data['filename'][i],image_file)
                        
                    ssd_labels = 'validation_ssd/labels/'
                    ssd_labels_2 = os.path.join(ssd_labels + row[1] +'_' + row[2].split('.')[0] + '_ssd.txt')
                    ssd_images = 'validation_ssd/images/'
                    ssd_images_2 = os.path.join(ssd_images + row[1] +'_' + row[2])
                    mode = 'a+' if os.path.isfile('test.txt') else 'w+'
                    trainval = open('test.txt', mode)
                    trainval.write("{} {}".format(ssd_images_2,ssd_labels_2))
                    trainval.write("\n")
                    trainval.close()
                    
                    #Create test_size file
                    mode = 'a+' if os.path.isfile('test_name_size.txt') else 'w+'
                    testsize = open('test_name_size.txt', mode)
                    imgtxt = row[2].split('.')[0].split('_')[1]
                    testsize.write("{} {} {}".format(imgtxt,height,width))
                    testsize.write("\n")
                    testsize.close()

print "Parsing Complete"
