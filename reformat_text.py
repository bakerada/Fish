import pandas as pd
import os
import sys


df = pd.read_csv(sys.argv[1],header=None)
df[0] = df[0].str.replace(sys.argv[2],"")
df.to_csv(sys.argv[1],header=None,index=False)
