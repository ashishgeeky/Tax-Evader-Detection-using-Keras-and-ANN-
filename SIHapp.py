from flask import Flask, flash, redirect, render_template, request, session, abort, Markup
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from keras import backend as K
from werkzeug import secure_filename
from sklearn.preprocessing import StandardScaler
import json
import time

app = Flask(__name__)
app.secret_key = os.urandom(12)

dropdown_list = []
dropdown_list_2 = []

def sav_name(n):
    with open("filename.txt", "w") as ff:
        ff.write(n)

def preprocess_data():
    ffr = open("filename.txt", "r")
    dataset = pd.read_csv('train.csv')
    upl_file = ffr.read()
    upl_file = str(upl_file)
    df_test = pd.read_csv(upl_file)
    X_test = df_test.iloc[:, 1:15].values
    X = dataset.iloc[:, 1:15].values

    from sklearn.preprocessing import LabelEncoder, StandardScaler

    X_train=X

    labelencoder_X_train_1 = LabelEncoder()
    X_train[:, 1] = labelencoder_X_train_1.fit_transform(X_train[:, 1])
    labelencoder_X_train_2 = LabelEncoder()
    X_train[:, 2] = labelencoder_X_train_2.fit_transform(X_train[:, 2])
    labelencoder_X_train_4 = LabelEncoder()
    X_train[:, 4] = labelencoder_X_train_4.fit_transform(X_train[:, 4])
    labelencoder_X_train_7 = LabelEncoder()
    X_train[:, 7] = labelencoder_X_train_7.fit_transform(X_train[:, 7])

    #labelencoder_X_test_1 = LabelEncoder()
    X_test[:, 1] = labelencoder_X_train_1.transform(X_test[:, 1])
    #labelencoder_X_test_2 = LabelEncoder()
    X_test[:, 2] = labelencoder_X_train_2.transform(X_test[:, 2])
    #labelencoder_X_test_4 = LabelEncoder()
    X_test[:, 4] = labelencoder_X_train_4.transform(X_test[:, 4])
    #labelencoder_X_test_7 = LabelEncoder()
    X_test[:, 7] = labelencoder_X_train_7.transform(X_test[:, 7])

    
    

    
    #from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_test

@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return render_template('upload.html')

@app.route('/login', methods=['GET', 'POST'])
def do_admin_login():
    error = None
    if request.form['username'] != 'dhfl' or request.form['password'] != 'dhfl':
        error = 'Invalid username or password. Please try again!'
    else:
        session['logged_in'] = True
        return home()

    return render_template('login.html', error=error)

@app.route("/logout")
def logout():
    session['logged_in'] = False
    session.clear()
    if os.path.exists('final_ans.csv'):
        os.remove('final_ans.csv')
    return home()

@app.route('/uploader', methods=['GET', 'POST'])
def uploader_file():
   if request.method == 'POST':
      K.clear_session()
      f = request.files['file']
      f.save(secure_filename(f.filename))
      sav_name(f.filename)

      ffr = open("filename.txt", "r")
      upl_file = ffr.read()
      upl_file = str(upl_file)
      
      test = preprocess_data()
      model = load_model('my_model.h5')
      model._make_predict_function()
      y_pred = model.predict(test)
      dff = pd.read_csv(upl_file)
      dff['Evaded Tax'] = y_pred
      #dff.set_index()
      #dff.sort_values('Exited', ascending=False, inplace=True)
      dff.to_csv('final_ans.csv',index=False) 
      return render_template('mytemplate3.html', data = dff.to_html(index=False,classes="data",table_id="mytable") )

@app.route('/uploader/individual',methods = ["POST"])
def individual_data():
    if request.method == "POST":
        data = request.form.to_dict()
        print(data)
        x = pd.DataFrame(data=data, index=[0])
        ls = ['age', 'fam', 'vehicles', 'houses','area', 'mtf', 'gold','bdfd','income', 'cibil']
        for i in ls:
            x[i][0] = int(x[i][0])
        
        #d = pd.DataFrame(data={'name': 'Arnab', 'age': 43, 'gender': 'Male', 'city': 'Kolkata', 'fam': 4, 'prof': 'Businessman', 'vehicles': 1, 'houses': 1, 'car': 'Sedan', 'area': 2000, 'mtf': 800000, 'gold': 30000, 'bdfd': 1078320
        #                      , 'income':1078320, 'cibil': 400},index=[0])
        x.to_csv('indvtest.csv',index=False)
        sav_name("indvtest.csv")

        test = preprocess_data()
        model = load_model('my_model.h5')
        model._make_predict_function()
        y_pred = model.predict(test)
        dff = pd.read_csv('indvtest.csv')
        dff['Evaded Tax'] = y_pred

        dff.to_csv('final_ans.csv', index=False)
        return render_template('mytemplate3.html', data = dff.to_html(index=False,classes="data",table_id="mytable") )




if __name__ == "__main__" :
    app.run(debug=True)
