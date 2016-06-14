from flask import Flask
import os

## Create app
app = Flask(__name__)

## Look up which is the current path
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

## Specify the upload directory target
app.config['UPLOAD_FOLDER'] = os.path.join(APP_ROOT, 'dynamic')

## Get the views
from webapp import views