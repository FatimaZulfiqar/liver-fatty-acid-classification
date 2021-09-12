from flask import Flask,request, url_for, redirect, render_template, jsonify
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
# set path to model file here
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

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    enc = LabelEncoder()
    encoded_user_input = data_unseen.copy()
    for col in category_col:
        encoded_user_input[col] = enc.fit_transform(encoded_user_input[col])

    pred = model.predict(encoded_user_input)
    print("\npredicted label: ",pred)
    if pred == 0:
        prediction = "S0 = 10% hepatic steatosis"
    elif pred == 1:
        prediction = "S1 = 11% to 33% hepatic steatosis"
    elif pred == 2:
        prediction = "S2 = 34% to 66% hepatic steatosis"
    elif pred == 3:
        prediction = "S3 = 67% or more hepatic steatosis"
    #prediction = int(prediction.Label[0])
    return render_template('home.html',pred='CAP Grade is  {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data],columns = cols)
    enc = LabelEncoder()
    encoded_user_input = data_unseen.copy()

    for col in category_col:
        encoded_user_input[col] = enc.fit_transform(encoded_user_input[col])
    prediction =model.predict(encoded_user_input)
    if prediction == 0:
        label = "S0 = 10% hepatic steatosis"
    elif prediction == 1:
        label = "S1 = 11% to 33% hepatic steatosis"
    elif prediction == 2:
        label = "S2 = 34% to 66% hepatic steatosis"
    elif prediction == 3:
        label = "S3 = 67% or more hepatic steatosis"
    # output = label
    output = 'CAP Grade is {}'.format(label)
    return jsonify(output)
if __name__ == '__main__':
    app.run(debug=True)
