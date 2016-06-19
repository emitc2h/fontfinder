## Python 3-like
from __future__ import absolute_import, division, print_function

## Image processing
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import cv2
import os
import cPickle as pickle

## My functions
from webapp.core.engine import connect_to_db
from webapp.core.utils import generate_letter_image

## ============================================
## Connect to DB
connection = connect_to_db()

query = '''
        SELECT DISTINCT aws_bucket_key FROM font_metadata ORDER BY aws_bucket_key;
        '''

## Retrieve data
df_font_metadata = pd.read_sql_query(query,connection)

n_fonts = len(df_font_metadata)

## Characters to generate
chars = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
    'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

local_path = os.path.join('/', 'home', 'ubuntu', 'fontfinder', 's3')

## --------------------------------------------
## First, create font->index map
index = 0
font_index_map = {}

for i,row in df_font_metadata.iterrows():
    font_index_map[row['aws_bucket_key']] = index
    index += 1

pickle.dump( font_index_map, open(os.path.join(local_path, 'font_index_map.p'), 'wb'))



## --------------------------------------------
## Second, create the numpy arrays
for char in chars:

    print('Now making array for {0} ...'.format(char))

    j=0
    arrays = []

    for i,row in df_font_metadata.iterrows():

        if j%10000==0: print('{0}/{1} done'.format(j, n_fonts))
        j+=1
    
        aws_bucket_key = row['aws_bucket_key']
        local_img_dir   = os.path.join(local_path, aws_bucket_key.split('.')[0])

        img_path = '{0}/{1}.jpg'.format(local_img_dir, char)
        if not os.path.exists(img_path): continue

        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((48,48), dtype='uint8')
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        arrays.append(img)

    np.save(os.path.join(local_path, char), np.stack(arrays))

print('All Done.')