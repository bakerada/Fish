import pandas as pd
import os
import glob
import sys

print sys.argv[1]
directory = os.path.join(os.getcwd(),sys.argv[1])
print directory
if not os.path.isdir(directory):
    raise ValueError('Directory does not exist')

files = glob.glob(directory + '*.jpg')
df = pd.DataFrame(files)
df.to_csv('train_boxes_ssd.csv',index=False,header=None)

