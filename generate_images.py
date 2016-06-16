## Python 3-like
from __future__ import absolute_import, division, print_function

## Image processing
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import cv2
import os

## My functions
from webapp.core.engine import connect_to_db
from webapp.core.utils import generate_letter_image

## ============================================
## Connect to DB
connection = connect_to_db()

query = '''
        SELECT * FROM font_metadata;
        '''

## Retrieve data
df_font_metadata = pd.read_sql_query(query,connection)

## Characters to generate
chars = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
    'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

local_path = os.path.join('/', 'home', 'ubuntu', 'fontfinder', 's3')

for i,row in df_font_metadata.iterrows():
    
    aws_bucket     = row['aws_bucket']
    aws_bucket_key = row['aws_bucket_key']
    
    local_font_file = os.path.join(local_path, aws_bucket_key)
    local_img_dir   = os.path.join(local_path, aws_bucket_key.split('.')[0])

    try:
        os.mkdir(local_img_dir)
    except OSError:
        pass

    if os.path.exists('{0}/{1}.jpg'.format(local_img_dir, chars[-1])):
        continue

    print('Now generating images for', aws_bucket_key)
    
    for char in chars:

        img_path = '{0}/{1}.jpg'.format(local_img_dir, char)

        if os.path.exists(img_path): continue
        
        img = np.zeros((48,48), dtype='uint8')
            
        try:
            img = generate_letter_image(char, local_font_file, imgsize=48)
        except:
            pass
            
        cv2.imwrite(img_path, img)

print('All Done.')