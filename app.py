import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler once
with open(r'Model/Model.pkl', 'rb') as model_pkl:
    model = pickle.load(model_pkl)


# Prediction function
def predict_loan_status(data):
    # Predict using the loaded model
    prediction = model.predict(data)
    return prediction

# Main application
def main():
    st.set_page_config(page_title="Loan Status Prediction App", layout="centered")

    # Front-end UI setup
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Loan Status Prediction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.sidebar.title("Enter Loan Details Below")

    # Collecting user input from the sidebar
    no_of_dependents = st.sidebar.number_input('Number of Dependents', min_value=0, max_value=10, step=1)
    self_employed = st.sidebar.selectbox('Self Employed', options=[0, 1], help="1 if self-employed, 0 otherwise")
    income_annum = st.sidebar.number_input('Annual Income', min_value=0, step=1000)
    loan_amount = st.sidebar.number_input('Loan Amount', min_value=0, step=1000)
    loan_term = st.sidebar.number_input('Loan Term (Months)', min_value=1, max_value=360, step=1)
    cibil_score = st.sidebar.number_input('CIBIL Score', min_value=300, max_value=900, step=1)
    residential_assets_value = st.sidebar.number_input('Residential Assets Value', min_value=0, step=1000)
    commercial_assets_value = st.sidebar.number_input('Commercial Assets Value', min_value=0, step=1000)
    luxury_assets_value = st.sidebar.number_input('Luxury Assets Value', min_value=0, step=1000)
    bank_asset_value = st.sidebar.number_input('Bank Assets Value', min_value=0, step=1000)

    # Prediction button
    if st.sidebar.button("Predict"):
        # Prepare the input data as an array
        data = np.array([[no_of_dependents, self_employed, income_annum, loan_amount, loan_term, cibil_score,
                          residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value]])
        
        # Standardize the input using the pre-fitted StandardScaler
        ss = StandardScaler()
        data = ss.fit_transform(data)
        
        # Make prediction
        result = predict_loan_status(data)

        # Display the result
        if result[0] == 1:
            st.success("Loan Approved!")
        else:
            st.error("Loan Denied.")

if __name__ == '__main__':
    main()
