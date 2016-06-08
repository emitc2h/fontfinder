from flask import render_template, request, flash, redirect, url_for
from webapp import app

import os

from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

## ----------------------------------------------
def allowed_file(filename):
    """
    Checks if the file extension is allowed
    """
    return ('.' in filename) and (filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS)


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
## ==============================================
def index():
    """
    Index page
    """

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

        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return render_template("starter-template.html", char_upload=url_for('static', filename=os.path.join('img', filename)))

    return render_template("starter-template.html", char_upload=url_for('static', filename='img/Mtest.png'))


@app.route('/upload', methods=['GET', 'POST'])
## ==============================================
def upload_file():
    """
    A page that uploads a file
    """
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

        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return redirect(url_for('upload_file', filename=filename))

    return '''
            <!doctype html>
            <title>Upload new File</title>
            <h1>Upload new File</h1>
            <form action="" method=post enctype=multipart/form-data>
              <p><input type=file name=file>
                <input type=submit value=Upload>
            </form>
           '''


