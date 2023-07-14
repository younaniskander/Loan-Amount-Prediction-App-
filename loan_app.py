import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('final_model.pkl')
Inputs = joblib.load('Inputs.pkl')
# Define the prediction function
def prediction(Gender, Married, Dependents, Education, Self_Employed, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, total_income):
    test_df = pd.DataFrame({
        'Gender': [Gender],
        'Married': [Married],
        'Dependents': [Dependents],
        'Education': [Education],
        'Self_Employed': [Self_Employed],
        'LoanAmount': [LoanAmount],
        'Loan_Amount_Term': [Loan_Amount_Term],
        'Credit_History': [Credit_History],
        'Property_Area': [Property_Area],
        'total_income': [total_income]
    })
    result = model.predict(test_df)
    return result[0]

# Define the Streamlit app
def main():
    st.set_page_config(page_title='Loan Amount Prediction App', page_icon=':money_with_wings:')
    st.title('Loan Amount Prediction App')
    st.write('Please fill out the following details to get the predicted loan amount.')
    # Define the input fields
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Married = st.selectbox('Marital Status', ['Yes', 'No'])
    Dependents = st.selectbox('Number of Dependents', [0, 1, 2, 3])
    Education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    Self_Employed = st.selectbox('Self Employed', ['Yes', 'No'])
    LoanAmount = st.slider('Loan Amount', min_value=1, max_value=500, step=1)
    Loan_Amount_Term = st.slider('Loan Amount Term', min_value=1, max_value=30, step=1)
    Credit_History = st.selectbox('Credit History', [0.0, 1.0])
    Property_Area = st.selectbox('Property Area', ['Rural', 'Semiurban', 'Urban'])
    total_income = st.slider('Total Income', min_value=1, max_value=100000, step=1)
    # Make the prediction
    if st.button('Predict'):
        with st.spinner('Predicting...'):
            result = prediction(Gender, Married, Dependents, Education, Self_Employed, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, total_income)
            st.success(f'The predicted loan amount is {result:.2f} INR')

if __name__ == '__main__':
    main()
