from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy import signal
import cv2

import os, math, random

import boto
from boto.s3.key import Key
from cStringIO import StringIO

## -------------------------------------
def read_img(path):
    """
    A function to read an image, and return a numpy array
    """

    # pil_img = Image.open(path)
    # npy_img = np.array(pil_img.getdata())
    # npy_img.shape = (pil_img.size[1], pil_img.size[0], npy_img.shape[1])

    ## Detect file extension, to orient the image
    ext = path.rsplit('.', 1)[1].lower()
    if ext in ['jpg', 'jpeg']:
        return np.rot90(np.asarray(Image.open(path)), -1).astype('uint8')[:,:,0:3]
    else:
        return np.asarray(Image.open(path)).astype('uint8')[:,:,0:3]




## -------------------------------------
def read_img_from_s3(bucket_key):
    """
    Returns an image read from s3 bucket
    """

    local_path = os.path.join('/', 'home', 'ubuntu', 'fontfinder', 's3')

    ## Try reading locally first
    try:
        pil_img = Image.open(os.path.join(local_path, bucket_key))

    ## Try reading on S3 bucket directly
    except IOError:
        return np.zeros((48,48), dtype='uint8')
        ## Connect to S3, get bucket (Assumes credentials exist in ~/.boto)
        # s3 = boto.connect_s3()
        # bucket = s3.get_bucket('fontfinder-fontfiles', validate=False)

        # ## Create and assign key
        # k = Key(bucket)
        # k.key = bucket_key

        # assert k.exists(), 'Image does not exist on fontfinder-fontfiles S3 bucket'

        # ## Get image as string, convert to file, load with PIL
        # s = k.get_contents_as_string()
        # pil_img = Image.open(StringIO(s))

    ## Convert to numpy array and return
    return np.array(pil_img.getdata()).reshape(48, 48)



## -------------------------------------
def bw_img(img):
    """
    Returns black and white image
    """

    if len(img.shape) > 2 and img.shape[2] > 1:

        variance = 0
        channel_with_max_variance = 0
        for i in range(img.shape[2]):
            current_variance = np.std(img[:,:,i])
            if current_variance > variance:
                variance = current_variance
                channel_with_max_variance = i

        current_variance = np.std(np.mean(img, axis=2))
        if current_variance > variance:
            bwing = np.mean(img, axis=2)
        else:
            bwimg = img[:,:,channel_with_max_variance]


        bwimg = np.mean(img, axis=2)
    else:
        bwimg = img

    return bwimg.astype('uint8')




## -------------------------------------
def bilateral_filter(img):
    """
    Applies bilateral filtering
    """

    return cv2.bilateralFilter(img, 15, 31, 31)




## -------------------------------------
def histogram(img):
    """
    Returns a histogram of tones in the image
    """

    return np.histogram(img, bins=64, range=[0,255])[0]




## --------------------------------------
def threshold(img):
    """
    Finds tonal peaks and return the bin indices of the two largest
    """

    hist_values = histogram(img)

    ## Creates a gaussian filter to smooth the array
    gfilter = signal.gaussian(11, std=1.7)
    smooth_values = np.convolve(hist_values, gfilter, mode='same')

    ## Collect maxima
    maxima        = signal.argrelextrema(smooth_values, np.greater)[0]
    maxima_values = smooth_values[maxima]

    if len(maxima) > 1:
        sorted_maxima = sorted(zip(maxima, maxima_values), key=lambda e: e[1], reverse=True)

        t = np.mean([sorted_maxima[0][0], sorted_maxima[1][0]])
        return int(t*4)
    else:
        return int(np.mean(img))




## --------------------------------------
def thresholding(img):
    """
    finds a good threshold and apply to image
    """

    t = threshold(img)

    print t

    background_tone = np.mean(
        [
            np.mean(img[0,:]),
            np.mean(img[-1,:]),
            np.mean(img[:,0]),
            np.mean(img[:,-1])
        ]
    )

    if background_tone < t:
        background_indices = img < t
        char_indices       = img > t
    else:
        background_indices = img > t
        char_indices       = img < t

    threshold_img = np.copy(img)

    threshold_img[background_indices] = 0
    threshold_img[char_indices]       = 255

    return threshold_img




## --------------------------------------
def edge_and_crop(img, imgsize=100):
    """
    Detects the edges of the character, rescale
    and produce a square image of the required size
    """
    ## Detect letter edges and crop
    x_projection = img.sum(axis=1)
    x_values = np.nonzero(x_projection)
    x0, x1 = x_values[0][0], x_values[0][-1]
    
    y_projection = img.sum(axis=0)
    y_values = np.nonzero(y_projection)
    y0, y1 = y_values[0][0], y_values[0][-1]
    
    img_crop = img[max(0, x0-3):min(x1+1, img.shape[0]-1), max(0, y0-3):min(y1+1, img.shape[1])]

    scaling_factor = 0.9*imgsize/max(img_crop.shape)
    new_h = int(img_crop.shape[0]*scaling_factor)
    new_w = int(img_crop.shape[1]*scaling_factor)

    img_crop  = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    ## Pad the image such that it fits, centered in a 100x100 px image
    array = np.zeros((imgsize,imgsize), dtype='float32')
    
    x_margin = imgsize - img_crop.shape[1]
    y_margin = imgsize - img_crop.shape[0]
    
    x0 = (x_margin//2 + x_margin%2) - 1
    x1 = (imgsize-1) - x_margin//2
    
    y0 = (y_margin//2 + y_margin%2) - 1
    y1 = (imgsize-1) - y_margin//2
    
    array[y0:y1, x0:x1] = img_crop
    
    return array




## --------------------------------------
def preprocess(path, imgsize=100):
    """
    A function to prepare the image to be read by the ConvNet
    """

    img = read_img(path)
    img = bw_img(img)
    img = bilateral_filter(img)
    img = thresholding(img)
    img = edge_and_crop(img, imgsize=imgsize)
    return img




## --------------------------------------
def save(img, path):
    """
    Save a file of a given name to a path
    """

    cv2.imwrite(path, img)




## -------------------------------------------
def generate_letter_image(letter, font_path, imgsize=100):
    """
    Generate a 100x100 grayscale image of a given letter
    of a given font
    """
    
    ## Test that there is a single character provided
    assert len(letter) == 1, 'The letter provided should contain exactly one alphanumeric character'
    
    ## Test that the character is alphanumeric
    assert letter.isalnum(), 'The letter provided should be alphanumeric'
    
    ## Make sure the entire character is contained in the raw image
    delta_x      = 0
    delta_y      = 0
    font_size    = imgsize*4
    
    left_check   = True
    right_check  = True
    top_check    = True
    bottom_check = True
    
    raw_w = raw_h = imgsize*10

    counter = 0

    while True:

        if counter > 10: break

        counter += 1
        
        ## Generate the image using PIL
        image_raw = Image.new(
            "L",                     ## single 8-bit channel
            (raw_w, raw_h),              ## 500x500 px
            0                        ## black background
        )
    
        ## Retrieve font
        font = ImageFont.truetype(font_path, font_size)
    
        ## Handle to draw on image
        draw = ImageDraw.Draw(image_raw)
        width, height = draw.textsize(letter, font=font)

        try:
            draw.text((raw_w/2 - width/2 + delta_x, raw_h/2 - height/2 + delta_y), letter, 255, font=font)
        except IOError:
            font_size = int(font_size * 0.9)
            continue
    
        array_raw = np.array(image_raw.getdata()).reshape(raw_w, raw_h)
        
        ## Detect whether or not the character is fully contained,
        ## and determine appropriate action
        
        previous_left_check   = left_check
        previous_right_check  = right_check
        previous_top_check    = top_check
        previous_bottom_check = bottom_check
        
        left_check   = array_raw[:,0].sum()   == 0
        right_check  = array_raw[:,(raw_w-1)].sum() == 0
        top_check    = array_raw[0,:].sum()   == 0
        bottom_check = array_raw[(raw_h-1),:].sum() == 0
        
        all_check = array_raw.sum() > 0
        
        assert all_check, 'Nothing was drawn, make sure the font has a representation for the provided character'
        
        ## When the character is container, the array sum should be superior to 0,
        ## but the bounding box should sum up to 0
        if  left_check and \
            right_check and \
            top_check and \
            bottom_check:
            break
            
        ## Let's find out if the character is too big
        if ((not left_check) and (not right_check)) or ((not top_check) and (not bottom_check)):
            font_size = int(font_size * 0.9)
            continue
            
        ## Shift the character around
        if not left_check:
            if not previous_right_check:
                font_size = int(font_size * 0.9)
            delta_x += width/4
            continue
            
        if not right_check:
            if not previous_left_check:
                font_size = int(font_size * 0.9)
            delta_x -= width/4
            continue
            
        if not top_check:
            if not previous_bottom_check:
                font_size = int(font_size * 0.9)
            delta_y += height/4
            continue
            
        if not bottom_check:
            if not previous_top_check:
                font_size = int(font_size * 0.9)
            delta_y -= height/4
            continue
            
    ## Detect letter edges and crop
    x_projection = array_raw.sum(axis=1)
    x_values = np.nonzero(x_projection)
    x0, x1 = x_values[0][0], x_values[0][-1]
    
    y_projection = array_raw.sum(axis=0)
    y_values = np.nonzero(y_projection)
    y0, y1 = y_values[0][0], y_values[0][-1]
    
    array_crop = array_raw[max(0, x0-3):x1, max(0, y0-3):y1]

    array_crop = array_crop.astype(float)
    
    scaling_factor = 0.9*imgsize/max(array_crop.shape)
    new_h = int(array_crop.shape[0]*scaling_factor)
    new_w = int(array_crop.shape[1]*scaling_factor)
    
    array_crop = cv2.resize(array_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    ## Pad the image such that it fits, centered in a 100x100 px image
    array = np.zeros((imgsize,imgsize), dtype='uint8')
    
    x_margin = imgsize - array_crop.shape[1]
    y_margin = imgsize - array_crop.shape[0]
    
    x0 = (x_margin//2 + x_margin%2) - 1
    x1 = (imgsize-1) - x_margin//2
    
    y0 = (y_margin//2 + y_margin%2) - 1
    y1 = (imgsize-1) - y_margin//2
    
    array[y0:y1, x0:x1] = array_crop
    
    return array


## ----------------------------------------
def margins(img):
    """
    Make sure the margins are the same
    """

    imgsize = max(img.shape)

    ## Detect letter edges and crop
    x_projection = img.sum(axis=1)
    x_values = np.nonzero(x_projection)
    x0, x1 = x_values[0][0], x_values[0][-1]
    
    y_projection = img.sum(axis=0)
    y_values = np.nonzero(y_projection)
    y0, y1 = y_values[0][0], y_values[0][-1]
    
    img_crop = img[max(0, x0-3):x1, max(0, y0-3):y1]

    img_crop = img_crop.astype(float)
    
    scaling_factor = 0.9*imgsize/max(img_crop.shape)
    new_h = int(img_crop.shape[0]*scaling_factor)
    new_w = int(img_crop.shape[1]*scaling_factor)
    
    img_crop = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    ## Pad the image such that it fits, centered in a 100x100 px image
    array = np.zeros(img.shape)
    
    x_margin = imgsize - img_crop.shape[1]
    y_margin = imgsize - img_crop.shape[0]
    
    x0 = (x_margin//2 + x_margin%2) - 1
    x1 = (imgsize-1) - x_margin//2
    
    y0 = (y_margin//2 + y_margin%2) - 1
    y1 = (imgsize-1) - y_margin//2
    
    array[y0:y1, x0:x1] = img_crop

    return array




## ----------------------------------------
def scale_variations(img, scale_factors=[0.9, 0.8, 0.7, 0.6]):
    """
    Produce scaling variations on input image
    """
    
    output_images = []
    
    for sf in scale_factors:
        
        canvas = np.zeros(img.shape, dtype=img.dtype)
        
        new_shape = (int(img.shape[0]*sf), int(img.shape[1]*sf))
        
        img_crop  = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
        
        x_margin = img.shape[1] - new_shape[1]
        y_margin = img.shape[0] - new_shape[0]
    
        x0 = (x_margin//2 + x_margin%2) - 1
        x1 = (img.shape[1]-1) - x_margin//2
    
        y0 = (y_margin//2 + y_margin%2) - 1
        y1 = (img.shape[1]-1) - y_margin//2
    
        canvas[y0:y1, x0:x1] = img_crop
        
        output_images.append(canvas)
        
    return output_images



## ----------------------------------------
def skew_variations(img, vertical_shear=[0], horizontal_shear=[0]):
    """
    Produces skewing variations on input image
    """
    
    output_images = []
    
    for vs in vertical_shear:
        for hs in horizontal_shear:
            
            origin = np.float32([[0,0],[1,0],[0,1]])
            shear  = np.float32([[hs,vs],[hs,-vs],[-hs,vs]])
            
            x = img.shape[1]//2
            y = img.shape[0]//2
            
            absolufy = np.float32([[x,y],[x,y],[x,y]])
            
            transformed = origin + shear + absolufy

            M = cv2.getAffineTransform(origin+absolufy, transformed)
            output_images.append(margins(cv2.warpAffine(img,M,img.shape)))
    
    return output_images



## ----------------------------------------
def rotate_variations(img, angles=[]):
    """
    Produces rotation variations on input image
    """
    
    output_images = []
    
    for a in angles:
        
        M = cv2.getRotationMatrix2D((img.shape[0]//2,img.shape[1]//2),a,1)
        output_images.append(margins(cv2.warpAffine(img,M,img.shape)))
        
    return output_images



## ----------------------------------------
def pixbypix_similarity(img1, img2):
    """
    Determines the pixel by pixel similarity of two images,
    assumes images are float normalized
    """
    
    assert img1.shape == img2.shape, 'Images should be of the same size'
    
    return 1.0 - np.sum(np.absolute(np.subtract(img1, img2)))/(img1.shape[0]*img1.shape[1])



## ----------------------------------------
def pix_occupancy(img):
    """
    Determines the occupancy of white pixels for an image
    """

    if img.ptp() > 1.0:
        return np.sum(normalize(img))/(img.shape[0]*img.shape[1])
    else:
        return np.sum(img)/(img.shape[0]*img.shape[1])


## ----------------------------------------
def normalize(img):
    """
    Normalize an image to range 0.0-1.0 float32
    """

    ptp = np.ptp(img)
    if ptp > 0:
        return np.multiply(np.add(img, -np.min(img)), 1.0/np.ptp(img))
    else:
        return np.zeros(img.shape)



## ----------------------------------------
def generate_noise(imgsize=100):
    """
    Generate a square b&w noise image of a given size
    """
    return margins(np.random.random(size=(imgsize, imgsize)))




