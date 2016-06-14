import utils
from sktfnn.neuralnetwork import NeuralNetwork
from sktfnn.layer import Layer, ConvLayer, DropoutLayer

## PostGres DB
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import numpy as np
import cv2, os, random, string

d=48

## --------------------------------------
def connect_to_db():
    """
    Connect to the fonts database
    """

    dbname = 'fonts_db'
    user   = 'mtm'
    pswd   = ''

    engine     = create_engine('postgresql://%s:%s@localhost/%s'%(user,pswd,dbname))
    connection = psycopg2.connect(database = dbname, user = user)

    return connection




## ---------------------------------------
def get_fontpath_list(connection):
    """
    Get the list of font paths in the DB
    """

    query = '''
        SELECT DISTINCT local_path1 FROM font_metadata;
    '''

    query_results = pd.read_sql_query(query,connection)

    return list(query_results['local_path1'].values)




## ----------------------------------------
def train_net(char, img, n_random=10):
    """
    Trains the neural network
    """

    ## Connect to database 
    con   = connect_to_db()
    flist = get_fontpath_list(con)

    ## Generate the training dataset
    X_train, y_train = utils.generate_training_sample(char, img, flist, n_random)

    p1=2
    p2=2

    ## Specify the neural network configuration
    nn = NeuralNetwork(
        hidden_layers = [
            ConvLayer(
                img_size=(d,d),
                patch_size=(5,5),
                n_features=64,
                pooling='max',
                pooling_size=(p1,p1),
            ),
            # ConvLayer(
            #     img_size=(d/p1,d/p1),
            #     n_features=64,
            #     pooling='max',
            # ),
            # ConvLayer(
            #     img_size=(d/(p1*p2),d/(p1*p2)),
            #     n_features=128,
            #     pooling='max',
            # ),
            Layer(
                n_neurons=1024,
                activation='relu'
            )
        ],
        learning_algorithm='Adam',
        cost_function='log-likelihood',
        learning_rate=1e-4,
        early_stopping=False,
        stagnation=10,
        n_epochs=50,
        mini_batch_size=y_train.shape[0]//10
        )

    ## Fit the model
    nn.fit(X_train, y_train, verbose=True, val_X=X_train, val_y=y_train)

    return nn

## ----------------------------------------
def evaluate(char, img_path, upload_path, n_random=100):
    """
    train and evaluate the model
    """

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print 'Learning your {0} ...'.format(char)
    print '- '*25

    ## train the neural net
    nn = train_net(char, img, n_random)

    ## Retrieve full database
    df = pd.read_sql_query('SELECT DISTINCT name, url, licensing, local_path1 FROM font_metadata;', connect_to_db())

    scores = []
    images = []

    progress = 0
    complete = len(df)

    print '='*50
    print 'Ranking fonts ...'
    print '- '*25


    for font in df['local_path1']:

        if progress%100 == 0:
            print 'progress {0}/{1}'.format(progress, complete)

        try:
            image = utils.generate_letter_image(char, font, imgsize=d)
        except:
            image = np.ones((d,d), dtype='uint8')

        norm_image = np.multiply(image, 1.0/256)
        norm_image.shape = (1,d*d)

        images.append(image)

        score = nn.predict_proba(norm_image)[0][0]
        scores.append(score)

        progress += 1

    image_idx = range(len(images))

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

        cv2.imwrite(img_path, images[row['img_idx']])

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
    sorted_img_idx = df['img_idx'].values

    font_index = 0
    y_fonts = []

    for i in range(100):
        x_fonts = []
        for j in range(100):
            img = images[sorted_img_idx[font_index]]
            x_fonts.append(img)
            font_index += 1

        y_fonts.append(np.hstack(x_fonts))
        
    all_fonts = np.vstack(y_fonts)

    cv2.imwrite(os.path.join(upload_path, 'out.png'), all_fonts)

    return results








