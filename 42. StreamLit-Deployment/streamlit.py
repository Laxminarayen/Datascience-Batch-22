import streamlit as st
import pandas as pd 
import joblib

#st.write("Code Change!!")
#Load the model 
model = joblib.load("StreamLit-Deployment/pipe.pkl")

#Function to make predictions 
def predict(data):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return abs(round(prediction,2))

#Streamlit app
st.title("Car Selling Price Prediction")

#SSub heading for the app
st.subheader("Please enter the values:")

#Form for the user to get the data 
with st.form(key = "prediction_form"):
    car_name = st.text_input("Car Name")
    year = st.number_input("Year", min_value = 1900,max_value = 2024, step = 1)
    present_price = st.number_input("Current Prices (in Lakhs)",step = 0.01)
    kms_driven = st.number_input("KMS Driven",step = 1)
    fuel_type = st.selectbox("Fuel Type", ["Petrol",'Diesel',"CNG"])
    seller_type = st.selectbox("Seller Type",['Dealer','Individual'])
    trasmission = st.selectbox("Transmission Type", ['Manual',"Automatic"])
    owner = st.number_input("Owner",min_value = 1, step=1)
    submit_button = st.form_submit_button(label = 'Predict')
#Car_Name,Year,Selling_Price,Present_Price,Kms_Driven,Fuel_Type,Seller_Type,Transmission,Owner
#Handling submission:
if submit_button: 
    user_input = {
        "Car_Name":car_name, 
        "Year":year,
        "Present_Price":present_price,
        "Kms_Driven":kms_driven,
        "Fuel_Type":fuel_type,
        "Seller_Type":seller_type,
        "Transmission":trasmission,
        "Owner":owner
    }
    result = predict(user_input)
    st.success(f"The Predicted Price is: {result} Lakhs")
    
#File Upload Section
st.subheader("Please Upload a csv file for Bulk Predictions:")

upload_file = st.file_uploader("Choose a CSV File", type = ['csv','xlsx','xls'])

if upload_file is not None: 
    #Handle my csv now 
    if upload_file.name.endswith("csv"):
        df = pd.read_csv(upload_file)
    else: 
        df = pd.read_excel(upload_file)
    
    st.write("File Uploaded Successfully!")
    st.write(df.head())
    
    if st.button("Predict for uploaded File"):
        df['Predicted_Price'] = model.predict(df)
        st.write("Prediction Completed!")
        st.write(df.head())
        st.download_button(label = "Download Predictions as CSV",data = df.to_csv(index=False),file_name = "predictions.csv",mime = "text/csv")