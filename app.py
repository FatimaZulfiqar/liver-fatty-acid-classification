from flask import Flask,request, url_for, redirect, render_template, jsonify
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# set path to model files here
model = pickle.load(open('model/model_smote.pkl', 'rb'))
cols = ['gender', 'age', 'bmi', 'obesity class ', 'OCP ',
       'Significant alcohol use', 'Hypothyroidism', 'IVDU ', 'DM ', 'HIV',
       'Concomitant Stvenvatin ', 'HCV Genotype', 'AST', 'ALT', 'Platelets',
       'Apolipoprotein ', 'macroglobulin', 'gene polymorphism',
       'HbA1c', 'Triglycerides', 'Pure insulin', 'Metformin', 'DPP401']

category_col =['HCV Genotype', 'gene polymorphism']

@app.route('/')
def home():
    return render_template("home.html")

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 29)
    result = model.predict(to_predict)
    return result[0]

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        pred = ValuePredictor(to_predict_list)

        print(pred)
        if int(pred) == 0:
            prediction = "S0 = 10% hepatic steatosis"
        elif int(pred) == 1:
            prediction = "S1 = 11% to 33% hepatic steatosis"
        elif int(pred) == 2:
            prediction = "S2 = 34% to 66% hepatic steatosis"
        elif int(pred) == 3:
            prediction = "S3 = 67% or more hepatic steatosis"

        return render_template('home.html',pred='CAP Grade is  {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction =model.predict(data_unseen)
    output = prediction
    return jsonify(output)
if __name__ == '__main__':
    app.run(debug=True)
