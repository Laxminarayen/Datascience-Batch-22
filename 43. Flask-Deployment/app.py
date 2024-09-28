from flask import Flask, render_template, request
import pandas as pd 
#import pickle
import joblib
app = Flask(__name__)

#with open("pipe.pkl",'rb') as file: 
    #pipe = pickle.load(file)
    
model = joblib.load("pipe.pkl")

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/predict", methods = ['POST'])
def predict():
    test_df = pd.DataFrame([request.form])
    value = model.predict(test_df)[0]
    value = str(round(value,2))
    return render_template("predict.html",price = value+" Lakhs")
    

if __name__ == '__main__':
    app.run(debug = True)