import utils
from sktfnn.neuralnetwork import NeuralNetwork
from sktfnn.layer import Layer, ConvLayer, DropoutLayer

## PostGres DB
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import numpy as np
import cv2, os, random, string, math

## Interacting with AWS S3 bucket
import boto
from boto.s3.key import Key

d=48

## --------------------------------------
def connect_to_db():
    """
    Connect to the fonts database
    """

    host   = 'fontdbinstance.c9mwqfkzqqmh.us-west-2.rds.amazonaws.com:5432'
    dbname = 'fontdb'

    user = ''
    pswd = ''

    with open('db_credentials', 'r') as f:
        credentials = f.readlines()
        f.close()
    
        user = credentials[0].rstrip()
        pswd = credentials[1].rstrip()

    connection = psycopg2.connect(
        database=dbname,
        user=user,
        password=pswd,
        host=host.split(':')[0],
        port=5432)

    return connection




## ----------------------------------------
def train_net(char, img, char_dict, n_random=10):
    """
    Trains the neural network
    """

    ## Connect to database 
    con   = connect_to_db()

    ## Generate the training dataset
    X_train, y_train = utils.generate_training_sample(char, img, char_dict, n_random)

    p1=2
    p2=4

    ## Specify the neural network configuration
    nn = NeuralNetwork(
        hidden_layers = [
            ConvLayer(
                img_size=(d,d),
                patch_size=(5,5),
                n_features=32,
                pooling='max',
                pooling_size=(p1,p1),
            ),
            ConvLayer(
                img_size=(d/p1,d/p1),
                n_features=64,
                pooling='max',
                pooling_size=(p2,p2)
            ),
            # ConvLayer(
            #     img_size=(d/(p1*p2),d/(p1*p2)),
            #     n_features=128,
            #     pooling='max',
            # ),
            Layer(
                n_neurons=128,
                activation='relu'
            )
        ],
        learning_algorithm='Adam',
        cost_function='log-likelihood',
        learning_rate=8e-3,
        target_accuracy=0.99,
        n_epochs=50,
        mini_batch_size=y_train.shape[0]
        )

    ## Fit the model
    nn.fit(X_train, y_train, verbose=True, val_X=X_train, val_y=y_train)

    return nn

## ----------------------------------------
def evaluate(char, img_path, upload_path, char_dict, n_random=10):
    """
    train and evaluate the model
    """

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print 'Learning your {0} ...'.format(char)
    print '- '*25

    ## train the neural net
    nn = train_net(char, img, char_dict, n_random)

    ## Retrieve full database
    df = pd.read_sql_query('SELECT DISTINCT name, url, licensing, aws_bucket_key FROM font_metadata;', connect_to_db())

    scores = []

    progress = 0
    complete = len(df)

    print '='*50
    print 'Ranking fonts ...'
    print '- '*25


    for font in df['aws_bucket_key']:

        if progress%1000 == 0:
            print 'progress {0}/{1}'.format(progress, complete)

        try:
            image = char_dict[font]
        except KeyError:
            image = np.zeros((d,d))

        norm_image = utils.normalize(image)
        norm_img   = utils.normalize(img)

        if utils.pixbypix_similarity(norm_img, norm_image) < 0.85:
            score = 0
        else:
            norm_image.shape = (1,d*d)
            score = nn.predict_proba(norm_image)[0][0]

        scores.append(score)

        progress += 1

    image_idx = range(len(scores))

    df['score']   = scores
    df['img_idx'] = image_idx

    df.sort_values('score', ascending=False, inplace=True)

    results = []

    top20 = df.head(20)

    i = 0

    for idx, row in top20.iterrows():

        random_string = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(10))

        img_name ='{0}.jpg'.format(random_string)
        img_path = os.path.join(upload_path, img_name)

        try:
            cv2.imwrite(img_path, char_dict[row['aws_bucket_key']])
        except KeyError:
            cv2.imwrite(img_path, np.zeros((48,48), dtype='uint8'))

        result = {
            'name'      : row['name'],
            'licensing' : row['licensing'],
            'url'       : row['url'],
            'origin'    : 'dafont.com',
            'img_path'  : img_name
            }

        results.append(result)

        i += 1

    ## Dump a grid image of the 10000 first results
    # sorted_img_idx = df['img_idx'].values
    # n_fonts = min(10000, len(sorted_img_idx))

    # font_index = 0
    # y_fonts = []

    # for i in range(int(math.sqrt(n_fonts))):
    #     x_fonts = []
    #     for j in range(int(math.sqrt(n_fonts))):
    #         img = images[sorted_img_idx[font_index]]
    #         x_fonts.append(img)
    #         font_index += 1

    #     y_fonts.append(np.hstack(x_fonts))
        
    # all_fonts = np.vstack(y_fonts)

    # cv2.imwrite(os.path.join(upload_path, 'out.png'), all_fonts)

    return results








