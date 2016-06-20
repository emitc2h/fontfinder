from flask import render_template, request, flash, redirect, url_for, send_from_directory, make_response, Response
from functools import wraps, update_wrapper
from datetime import datetime
from webapp import app, engine

from core import utils

import os, datetime, random, string, time
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
@app.route('/search', methods=['GET', 'POST'])
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
            img = utils.preprocess(save_path, imgsize=48)
            engine.get_user_image(random_fname, img)

            new_save_path = os.path.join(app.config['UPLOAD_FOLDER'], engine.user_image_path)
            utils.save(255 - img, new_save_path)

            print request.url

            r = make_response(
                    render_template(
                        "search.html",
                        char_upload=request.url_root+'dynamic/{0}'.format(engine.user_image_path),
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

            engine.get_user_char(character)
            y = engine.initialize()

            keep_going = True
            while(keep_going):
                accuracy, keep_going = engine.iteration(y)
            engine.prepare_evaluation()
            for i in range(100):
                print i
                engine.evaluate_one_percent()
            results = engine.finalize(app.config['UPLOAD_FOLDER'])

            i = 0
            grid_results = []
            while i < len(results):
                grid_results.append(results[i:i+4])
                i+=4

            r = make_response(
                    render_template(
                        "search.html",
                        char_upload=request.url_root+'dynamic/{0}'.format(engine.user_image_path),
                        char_placeholder=char_placeholder,
                        results=grid_results
                    )
                )

            return add_headers(r)

        else:
            r = make_response(
                    render_template(
                        "search.html",
                        char_upload=url_for('static', filename='img/M.png'),
                        char_placeholder=char_placeholder,
                        results=[]
                    )
                )

            return add_headers(r)




@app.route('/tips-and-tricks')
## ==============================================
def tips_and_tricks():
    """
    tips and tricks page
    """

    return render_template('tips-and-tricks.html')




@app.route('/about')
## ==============================================
def about():
    """
    about page
    """

    return render_template('about.html')




@app.route('/contact')
## ==============================================
def contact():
    """
    tips and tricks page
    """

    return render_template('contact.html')




@app.route('/progress')
## ==============================================
def progress():
    def generate():
        x = 0
        while x < 100:
            print x
            x += 10
            time.sleep(1)
            yield 'data:{0}\n\n'.format(x)
    return Response(generate(), mimetype='text/event-stream')















