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

for char in chars:

    print('Now making pickle for {0} ...'.format(char))

    char_dict = {}

    j=0

    for i,row in df_font_metadata.iterrows():

        if j%1000==0: print('{0}/{1} done'.format(j, n_fonts))
        j+=1
    
        aws_bucket_key = row['aws_bucket_key']
        local_img_dir   = os.path.join(local_path, aws_bucket_key.split('.')[0])

        img_path = '{0}/{1}.jpg'.format(local_img_dir, char)
        if not os.path.exists(img_path): continue

        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            continue
            
        char_dict[aws_bucket_key] = img

    pickle.dump( char_dict, open(os.path.join(local_path, 'newps/{0}.p'.format(char)), 'wb') )

print('All Done.')