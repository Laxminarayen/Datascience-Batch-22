#Container - lightweight - Stand-alone - executable 
#Image - A read-only template - Use this to create a container
#DockerScript - Set of instructions on how to build a docker image..
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("pipe.pkl")

@app.route("/")
def home_page():
    return render_template("index.html")

# Ensure categorical values are properly formatted
def preprocess_input(input_data):
    # Ensure the correct format for categorical columns
    input_data['Fuel_Type'] = input_data['Fuel_Type'].str.capitalize()  # Capitalize fuel type
    input_data['Seller_Type'] = input_data['Seller_Type'].str.capitalize()  # Capitalize seller type
    input_data['Transmission'] = input_data['Transmission'].str.capitalize()  # Capitalize transmission
    return input_data

@app.route("/predict", methods=['POST'])
def predict():
    # Collect data from the form and create a DataFrame
    test_df = pd.DataFrame([request.form])
    
    # Preprocess the input to ensure it's in the correct format
    test_df = preprocess_input(test_df)

    # Make a prediction
    value = model.predict(test_df)[0]
    value = str(round(value, 2))

    return render_template("predict.html", price=value + " Lakhs")

if __name__ == '__main__':
    app.run(debug=True)
