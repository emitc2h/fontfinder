from flask import render_template, request, flash, redirect, url_for, send_from_directory, make_response
from functools import wraps, update_wrapper
from datetime import datetime
from webapp import app

from core import utils, engine

import os, datetime, random, string
import cPickle as pickle

from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

usr_upload_fnames = []

## ----------------------------------------------
def allowed_file(filename):
    """
    Checks if the file extension is allowed
    """
    return ('.' in filename) and (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


## ----------------------------------------------
def add_headers(response):
    """
    Makes sure there's no image caching
    """

    response.headers.add('Last-Modified', datetime.datetime.now())
    response.headers.add('Cache-Control', 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0')
    response.headers.add('Pragma', 'no-cache')

    return response

@app.route('/dynamic/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
## ==============================================
def index():
    """
    Index page
    """

    char_placeholder='M'

    ## - - - - - - - - - - - - - - - - - - - - - - - - 
    if request.method == 'POST':

        ## Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        f = request.files['file']

        ## if user does not select file, browser also
        ## submit an empty part without filename
        if f.filename == '':
            flash('No selected file')
            return redirect(request.url)

        ## If the right kind of file is uploaded
        if f and allowed_file(f.filename):

            ## Get uploaded file extension
            ext = f.filename.rsplit('.', 1)[1]

            ## Rename file to usr_upload.<ext>
            filename = secure_filename('usr_upload.{0}'.format(ext))

            ## save to the upload folder (static/img)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(save_path)

            ## preprocess the image and save again
            random_fname = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5)) + '.jpg'
            usr_upload_fnames.append(random_fname)

            img = utils.preprocess(save_path, imgsize=48)
            new_save_path = os.path.join(app.config['UPLOAD_FOLDER'], usr_upload_fnames[-1])
            utils.save(img, new_save_path)

            print request.url

            r = make_response(
                    render_template(
                        "starter-template.html",
                        char_upload=request.url_root+'dynamic/{0}'.format(usr_upload_fnames[-1]),
                        char_placeholder=char_placeholder,
                        results=[]
                    )
                )

            return add_headers(r)



    ## - - - - - - - - - - - - - - - - - - - - - - - - 
    if request.method == 'GET':

        character = request.args.get('character')

        if character and len(character) == 1 and character.isalnum():

            char_placeholder = character

            local_path = os.path.join('/', 'home', 'ubuntu', 'fontfinder', 's3')

            char_dict = pickle.load( open( os.path.join(local_path, '{0}.p'.format(character)), 'rb' ) )

            results = engine.evaluate(
                character,
                os.path.join(app.config['UPLOAD_FOLDER'], usr_upload_fnames[-1]),
                app.config['UPLOAD_FOLDER'],
                char_dict
                )

            i = 0
            grid_results = []
            while i < len(results):
                grid_results.append(results[i:i+4])
                i+=4

            r = make_response(
                    render_template(
                        "starter-template.html",
                        char_upload=request.url_root+'dynamic/{0}'.format(usr_upload_fnames[-1]),
                        char_placeholder=char_placeholder,
                        results=grid_results
                    )
                )

            return add_headers(r)

        else:
            r = make_response(
                    render_template(
                        "starter-template.html",
                        char_upload=url_for('static', filename='img/M.png'),
                        char_placeholder=char_placeholder,
                        results=[]
                    )
                )

            return add_headers(r)
























