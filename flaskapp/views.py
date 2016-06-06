from flask import render_template
from flaskapp import app

@app.route('/')
@app.route('/index')

## ============================================
def index():
    """
    index.html
    """
    
    user = {'nickname' : 'mtm'}

    return render_template(
        'index.html',
        title='Home',
        user=user
    )