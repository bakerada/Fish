import json
import csv
import pandas as pd
import os
import glob
import shutil
import numpy as np

cwd = os.getcwd()
print cwd

# This should be run in directory that houses training data
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
if not os.path.isdir(training_images):
    os.makedirs(training_images)
if not os.path.isdir(training_labels):
    os.makedirs(training_labels)
if not os.path.isdir(validation_images):
    os.makedirs(validation_images)
if not os.path.isdir(validation_labels):
    os.makedirs(validation_labels)



jsons = glob.glob('*.json')

# SSD to be added
models = ['detectnet']

for model in models:
    for j in jsons:
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
        
        for i in range(n):
            row = data.iloc[i].values.tolist()
            # Traning set
            if i <= cut:
                if data['filename'][i] != data['filename'][np.minimum(i+1,n-1)]:
                    label_file = os.path.join(training_labels + row[1] +'_' + row[2].split('.')[0] + '.txt')
                    image_file = os.path.join(training_images + row[1] +'_' + row[2])
                    if model == 'detectnet':
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

            #validation set    
            else:
                label_file = os.path.join(validation_labels + row[1] +'_' + row[2].split('.')[0] + '.txt')
                image_file = os.path.join(validation_images + row[1] +'_' + row[2])
                if model == 'detectnet':
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
