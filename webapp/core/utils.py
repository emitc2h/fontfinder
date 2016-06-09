from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy import signal
from skimage import transform
import cv2

import os

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
        return np.rot90(np.asarray(Image.open(path)), -1)
    else:
        return np.asarray(Image.open(path)).astype('uint8')




## -------------------------------------
def bw_img(img):
    """
    Returns normalized black and white image
    """

    if len(img.shape) > 2 and img.shape[2] > 1:
        bwimg = np.mean(img, axis=2)
    else:
        bwimg = img

    return bwimg




## -------------------------------------
def bilateral_filter(img):
    """
    Applies bilateral filtering
    """

    return cv2.bilateralFilter(img[:,:,0:3], 15, 41, 41)




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
    gfilter = signal.gaussian(11, std=1.5)
    smooth_values = np.convolve(hist_values, gfilter, mode='same')

    ## Collect maxima
    maxima = signal.argrelextrema(smooth_values, np.greater)[0]

    ## Order maxima by size
    sorted_maxima = sorted(zip(maxima, smooth_values[maxima]), key=lambda e: e[1], reverse=True)

    if len(sorted_maxima) > 1:
        return np.mean([sorted_maxima[0][0], sorted_maxima[1][0]])*4
    else:
        return np.mean(img)




## --------------------------------------
def thresholding(img):
    """
    finds a good threshold and apply to image
    """

    t = threshold(img)

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
    
    img_crop = img[max(0, x0-3):x1, max(0, y0-3):y1]

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
    img = bilateral_filter(img)
    img = bw_img(img)
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
    font_size    = imgsize*5
    
    left_check   = True
    right_check  = True
    top_check    = True
    bottom_check = True
    
    raw_w = raw_h = imgsize*5

    while True:
        
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
    
    scaling_factor = 0.9*imgsize/max(array_crop.shape)
    
    array_crop = transform.rescale(array_crop, scale=scaling_factor, preserve_range=True)
    
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
def generate_noise(imgsize=100):
    """
    Generate a square b&w noise image of a given size
    """
    return np.random.random(size=(imgsize, imgsize))



## ----------------------------------------
def generate_training_sample(char, img, font_list, n_random=10):
    """
    A function to generate the training sample
    """
    
    ## normalize the image
    norm_img = np.multiply(np.add(img, -np.min(img)), 1.0/np.ptp(img))
    
    ## Get image dimensions
    w,h = norm_img.shape
    assert w == h, 'Char image should be square'
    
    ## Obtain random fonts
    random_fonts = np.random.choice(font_list, n_random)
    random = []

    for font in random_fonts:
        try:
            rdn_img      = generate_letter_image(char, font, imgsize=w)
            rdn_norm_img = np.multiply(rdn_img, 1.0/255)
            
            random.append(np.ravel(rdn_norm_img))
        except:
            random.append(np.ones(w*w))
            
    ## Put together the different types of training samples
    n_noise = 10
    noise   = [np.ravel(generate_noise(imgsize=w)) for i in range(n_noise)]
    
    n_zeros = 10
    zeros   = [np.zeros(w*w)]*n_zeros
    
    n_signal = n_random + n_noise + n_zeros
    signal   = [np.ravel(norm_img)]*n_signal
    
    n_total = 2*n_signal
    
    X = np.stack(signal + noise + zeros + random, axis=0)
    y = np.array([0]*n_signal + [1]*n_noise + [2]*n_zeros + range(3, n_random+3))
    
    return X,y




