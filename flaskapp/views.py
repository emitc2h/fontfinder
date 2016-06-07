from flask import render_template
from flask import request

from flaskapp import app

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

import pandas as pd

from a_Model import ModelIt

user = 'mtm'
host = 'localhost'
dbname = 'birth_db'
db = create_engine('postgres://{0}{1}/{2}'.format(user, host, dbname))

con = None
con = psycopg2.connect(database=dbname, user=user)



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



@app.route('/db')
## ============================================
def birth_page():
    """
    db.html
    """

    sql_query = """
                SELECT * FROM birth_data_table
                    WHERE delivery_method='Cesarean';
                """

    query_results = pd.read_sql_query(sql_query, con)
    births        = ''
    for i in range(10):
        births += query_results.iloc[i]['birth_month']
        births += '<br>'

    return births



@app.route('/db_fancy')
## ============================================
def cesareans_page_fancy():
    """
    db_fancy.html
    """

    sql_query = """
               SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';
                """
    query_results=pd.read_sql_query(sql_query,con)
    births = []
    for i in range(0,query_results.shape[0]):
        births.append(
            dict(
                index=query_results.iloc[i]['index'],
                attendant=query_results.iloc[i]['attendant'],
                birth_month=query_results.iloc[i]['birth_month']
                )
            )

    return render_template('cesareans.html',births=births)



@app.route('/input')
## ============================================
def cesareans_input():
    """
    input.html
    """
    return render_template("input.html")



@app.route('/output')
## ============================================
def cesareans_output():
    """
    output.html
    """

    #pull 'birth_month' from input field and store it
    patient = request.args.get('birth_month')

    #just select the Cesareans  from the birth dtabase for the month that the user inputs
    query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
    print query

    query_results=pd.read_sql_query(query,con)
    print query_results

    births = []
    for i in range(0,query_results.shape[0]):
        births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
        
    the_result = ModelIt(patient,births)

    return render_template("output.html", births=births, the_result=the_result)














