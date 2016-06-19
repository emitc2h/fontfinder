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
import cPickle as pickle

d=48
local_path = os.path.join('/', 'home', 'ubuntu', 'fontfinder')

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

    ## Generate the training dataset
    X_train, y_train = utils.generate_training_sample(char, img, char_dict, n_random)

    p1=2
    p2=2

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
                patch_size=(5,5),
                n_features=64,
                pooling='max',
                pooling_size=(p2,p2)
            ),
            Layer(
                n_neurons=512,
                activation='relu'
            )
        ],
        learning_algorithm='Adam',
        cost_function='log-likelihood',
        learning_rate=1e-3,
        target_accuracy=1.0,
        n_epochs=20,
        mini_batch_size=y_train.shape[0]//5
        )

    ## Fit the model
    nn.fit(X_train, y_train, verbose=True, val_X=X_train, val_y=y_train)

    return nn

## ----------------------------------------
def evaluate(char, img_path, upload_path, char_dict, n_random=200):
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

        if abs(utils.pix_occupancy(norm_img) - utils.pix_occupancy(norm_image)) > 0.10 or \
           utils.pixbypix_similarity(norm_img, norm_image) < 0.80:
            score = 0
        else:
            norm_image.shape = (1,d*d)
            pred = nn.predict_proba(norm_image)[0]
            score = pred[0]/np.mean(pred[1:])

        scores.append(score)

        progress += 1

    image_idx = range(len(scores))

    df['score']   = scores
    df['img_idx'] = image_idx

    df.sort_values('score', ascending=False, inplace=True)

    results = []

    top = df.head(32)

    i = 0

    for idx, row in top.iterrows():

        random_string = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(10))

        img_name ='{0}.jpg'.format(random_string)
        img_path = os.path.join(upload_path, img_name)

        try:
            cv2.imwrite(img_path, 255 - char_dict[row['aws_bucket_key']])
        except KeyError:
            cv2.imwrite(img_path, 255 - np.zeros((48,48), dtype='uint8'))

        result = {
            'name'      : row['name'],
            'licensing' : row['licensing'],
            'url'       : row['url'],
            'score'     : '{0:.0f}% match quality'.format(row['score']*100),
            'file'      : os.path.split(row['aws_bucket_key'])[-1],
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


## ==============================================
class Engine(object):
    """
    The class that drives the ConvNet learning and evaluating
    """

    ## ----------------------------------------------
    def __init__(self):
        """
        Constructor
        """

        ## Accessing character image data
        self.font_index_map = pickle.load(open(os.path.join(local_path, 's3', 'font_index_map.p'), 'rb'))

        ## Database connection
        self.dbhost = 'fontdbinstance.c9mwqfkzqqmh.us-west-2.rds.amazonaws.com:5432'
        self.dbname = 'fontdb'
        self.connection = None
        self.connect_to_db()

        ## Neural Network
        self.nn = None
        self.X  = None
        self.y  = None

        ## Training sample parameters
        self.n_random   = 200
        self.char_array = None


    ## --------------------------------------
    def connect_to_db(self):
        """
        Connect to the fonts database
        """

        user = ''
        pswd = ''

        with open(os.path.join(local_path, 'db_credentials'), 'r') as f:
            credentials = f.readlines()
            f.close()
    
        user = credentials[0].rstrip()
        pswd = credentials[1].rstrip()

        self.connection = psycopg2.connect(
            database=self.dbname,
            user=user,
            password=pswd,
            host=self.dbhost.split(':')[0],
            port=5432)


    ## --------------------------------------
    def load_char_array(self, char):
        """
        Loads the character array for a given character
        """

        self.char_array = np.load(os.path.join(local_path, 's3', '{0}.npy'.format(char)))


    ## ----------------------------------------
    def generate_training_sample(self, char, img):
        """
        A function to generate the training samples
        """
    
        random.seed(42)

        ## normalize the image
        norm_img = utils.normalize(img)
    
        ## Get image dimensions
        w,h = norm_img.shape
        assert w == h, 'Char image should be square'
    
        ## Obtain similar enough random fonts
        random_fonts = []

        n_fonts = self.char_array.shape[0]

        endloop = 0

        n_random = self.n_random

        while len(random_fonts) < n_random:

            rdn_img = self.char_array[random.randint(0, n_fonts)]

            rdn_norm_img = utils.normalize(rdn_img)
            pbp          = utils.pixbypix_similarity(rdn_norm_img, norm_img)
            if (pbp < 0.9999):
                random_fonts.append(np.ravel(rdn_norm_img))

            ## Bail out of the loop if not enough similar fonts are found
            if endloop > 20000:
                n_random = len(random_fonts)
                break

            endloop += 1

        print 'Found {0} fonts for the random sample'.format(n_random)

        ## Put together the different types of training samples    
        n_signal = n_random

        n_variations = n_signal//4

        variations = []
        variations += utils.scale_variations(norm_img, scale_factors=np.linspace(0.95, 0.99, n_variations))
        variations += utils.skew_variations(norm_img, vertical_shear=np.linspace(-0.02, 0.02, math.ceil(math.sqrt(n_variations))), horizontal_shear=np.linspace(-0.02, 0.02, math.ceil(math.sqrt(n_variations))))
        variations += utils.rotate_variations(norm_img, angles=np.linspace(-5,5, n_variations))
        variations += [norm_img]*n_variations

        signal = [np.ravel(var) for var in variations]

        self.X = np.stack(signal + random_fonts, axis=0)
        self.y = np.array([0]*len(signal) + range(1, n_random+1))


    ## --------------------------------------
    def initialize(self, char, img):
        """
        Initializes the neural net
        """

        ## Load the character array
        self.load_char_array(char)

        ## Generate the training sample
        self.generate_training_sample(char, img)

        ## Specify the neural network
        p1=2
        p2=2

        ## Specify the neural network configuration
        self.nn = NeuralNetwork(
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
                    patch_size=(5,5),
                    n_features=64,
                    pooling='max',
                    pooling_size=(p2,p2)
                ),
                Layer(
                    n_neurons=512,
                    activation='relu'
                )
            ],
            learning_algorithm='Adam',
            cost_function='log-likelihood',
            learning_rate=1e-3,
            target_accuracy=1.0,
            n_epochs=20,
            mini_batch_size=self.y.shape[0]//5
        )







