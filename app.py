from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import numpy as np  
from tensorflow.keras.models import load_model
import joblib



def return_prediction(model,scaler,sample_json):
    
    # For larger data features, you should probably write a for loop
    # That builds out this array for you
    
    cp_worst = sample_json['concave_points_worst']
    a_worst = sample_json['area_worst']
    r_worst = sample_json['radius_worst']
    p_worst = sample_json['perimeter_worst']
    cp_mean = sample_json['concave_points_mean']
    
    X = [[cp_worst,a_worst,r_worst,p_worst,cp_mean]]
    
    X_scaled = scaler.transform(X)
    
    classes = np.array(['Malignant', 'Benign'])
    
    class_ind = model.predict_classes(X_scaled)
    
    return classes[class_ind][0]



app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'someRandomKey'


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
cancer_dim_model = load_model("cancer_data_model.h5")
cancer_dim_scaler = joblib.load("cancer_data_scaler")


# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class CancerForm(FlaskForm):
    cp_worst = TextField('Concave Points Worst')
    a_worst = TextField('Area Worst ')
    r_worst = TextField('Radius Worst')
    p_worst = TextField('Perimeter Worst')
    cp_mean = TextField('Concave Points Mean')

    submit = SubmitField('Analyze')



@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = CancerForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        session['cp_worst'] = form.cp_worst.data
        session['a_worst'] = form.a_worst.data
        session['r_worst'] = form.r_worst.data
        session['p_worst'] = form.p_worst.data
        session['cp_mean'] = form.cp_mean.data

        return redirect(url_for("prediction"))


    return render_template('main.html', form=form)


@app.route('/prediction')
def prediction():

    content = {}

    content['concave_points_worst'] = float(session['cp_worst'])
    content['area_worst'] = float(session['a_worst'])
    content['radius_worst'] = float(session['r_worst'])
    content['perimeter_worst'] = float(session['p_worst'])
    content['concave_points_mean'] = float(session['cp_worst'])

    results = return_prediction(model=cancer_dim_model,scaler=cancer_dim_scaler,sample_json=content)

    return render_template('prediction.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)
