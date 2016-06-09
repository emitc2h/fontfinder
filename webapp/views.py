from flask import render_template, request, flash, redirect, url_for, send_from_directory, make_response
from functools import wraps, update_wrapper
from datetime import datetime
from webapp import app

from core import utils, engine

import os

from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

## ----------------------------------------------
def allowed_file(filename):
    """
    Checks if the file extension is allowed
    """
    return ('.' in filename) and (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


## ----------------------------------------------
def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
        
    return update_wrapper(no_cache, view)


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
@nocache
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
            img = utils.preprocess(save_path, imgsize=48)
            new_save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'usr_upload.jpg')
            utils.save(img, new_save_path)

            return render_template(
                "starter-template.html",
                char_upload=url_for('static', filename=os.path.join('img/usr_upload.jpg')),
                char_placeholder=char_placeholder,
                results=[]
                )


    ## - - - - - - - - - - - - - - - - - - - - - - - - 
    if request.method == 'GET':

        character = request.args.get('character')
        if character and len(character) == 1 and character.isalnum():

            char_placeholder = character

            results = engine.evaluate(
                character,
                os.path.join(app.config['UPLOAD_FOLDER'], 'usr_upload.jpg'),
                app.config['UPLOAD_FOLDER'],
                n_random=100
                )

            return render_template(
                "starter-template.html",
                char_upload=url_for('static', filename=os.path.join('img/usr_upload.jpg')),
                char_placeholder=char_placeholder,
                results=results
                )

        else:

            return render_template(
                "starter-template.html",
                char_upload=url_for('static', filename='img/M.png'),
                char_placeholder=char_placeholder,
                results=[]
                )
























