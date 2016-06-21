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

        ## user image
        self.user_image      = None
        self.user_image_path = None
        self.user_char       = None

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
        self.n_random   = 300
        self.char_array = None

        ## Evaluation and results
        self.df          = None
        self.df_percents = []
        self.percent     = 0
        self.scores      = {}
        self.n_fonts     = 0
        self.evaluated   = 0
        self.got_results = False


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
    def load_char_array(self):
        """
        Loads the character array for a given character
        """

        self.char_array = np.load(os.path.join(local_path, 's3', '{0}.npy'.format(self.user_char)))


    ## ----------------------------------------
    def generate_training_sample(self):
        """
        A function to generate the training samples
        """
    
        random.seed(42)

        ## normalize the image
        norm_img = utils.normalize(self.user_image)
    
        ## Get image dimensions
        w,h = norm_img.shape
        assert w == h, 'Character image should be square'
    
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
    def get_user_image(self, user_image_path, user_image):
        """
        obtain user image
        """

        self.user_image      = user_image
        self.user_image_path = user_image_path


    ## --------------------------------------
    def get_user_char(self, user_char):
        """
        obtain user image
        """

        self.user_char = user_char


    ## --------------------------------------
    def initialize(self):
        """
        Initializes the neural net
        """

        ## Load the character array
        self.load_char_array()

        ## Generate the training sample
        self.generate_training_sample()

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
            output_activation='softmax',
            cost_function='log-likelihood',
            learning_rate=1e-3,
            target_accuracy=0.99,
            n_epochs=20,
            mini_batch_size=self.y.shape[0]//5
        )

        y_one_hot, _ = self.nn.prepare_fit(self.X, self.y, verbose=True, val_X=self.X, val_y=self.y)

        return y_one_hot


    ## --------------------------------------
    def iteration(self, y_one_hot):
        """
        Do one iteration
        """

        return self.nn.epoch(self.X, y_one_hot, True, self.X, y_one_hot)



    ## --------------------------------------
    def prepare_evaluation(self):
        """
        Prepare the evaluation of the neural net on the fonts
        """

        ## Retrieve full database
        self.df          = pd.read_sql_query('SELECT DISTINCT name, url, licensing, aws_bucket_key FROM font_metadata ORDER BY aws_bucket_key;', self.connection)

        a = np.repeat(range(99), len(self.df)//100)
        b = np.repeat(99, len(self.df) - len(a))
        c = np.append(a,b)

        self.df['order_index'] = c
        self.df_percents = [self.df[self.df['order_index'] == i] for i in range(100)]
        self.percent     = 0
        self.scores      = {}
        self.n_fonts     = len(self.df)



    ## --------------------------------------
    def evaluate_one_percent(self):
        """
        evaluate one percent of all fonts
        """

        if self.percent >= len(self.df_percents):
            self.got_results = True
            return False

        for _,row in self.df_percents[self.percent].iterrows():

            font = row['aws_bucket_key']
            img  = self.char_array[self.font_index_map[font]]

            norm_img      = utils.normalize(img)
            norm_user_img = utils.normalize(self.user_image)

            if abs(utils.pix_occupancy(norm_img) - utils.pix_occupancy(norm_user_img)) > 0.10 or \
              utils.pixbypix_similarity(norm_img, norm_user_img) < 0.80:
                score = 0
            else:
                norm_img.shape = (1,d*d)
                pred = self.nn.predict_proba(norm_img)[0]
                score = pred[0]/np.max(pred[1:])

            self.scores[font] = score

        self.percent += 1

        return True



    ## --------------------------------------
    def finalize(self, upload_path):
        """
        Rank the fonts
        """

        self.df['score'] = self.df['aws_bucket_key'].map(self.scores)
        self.df.sort_values('score', ascending=False, inplace=True)

        results = []

        top = self.df.head(32)

        i = 0

        for _, row in top.iterrows():

            font = row['aws_bucket_key']
            img  = self.char_array[self.font_index_map[font]]

            random_string = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(10))

            img_name ='{0}.jpg'.format(random_string)
            img_path = os.path.join(upload_path, img_name)

            try:
                cv2.imwrite(img_path, 255 - img)
            except KeyError:
                cv2.imwrite(img_path, 255 - np.zeros((48,48), dtype='uint8'))

            result = {
                'name'      : row['name'],
                'licensing' : row['licensing'],
                'url'       : row['url'],
                'file'      : os.path.split(font)[-1],
                'origin'    : 'dafont.com',
                'img_path'  : img_name
                }

            results.append(result)

            i += 1

        self.got_results = False

        return results







