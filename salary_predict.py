import streamlit as st 
import pandas as pd 
st.title("🎈 My new app")
st.write(
     "Let's start building!"
)

first_name = st.text_input("First name")
last_name = st.text_input("Last name")
gender = st.number_input("Your age", 0, 100, 30, 1)
dob = st.date_input("Your birthday")
marital_status = st.radio("Marital status", ["Single", "Married"])
years_of_experience = st.slider("Years of experience", 0, 40)

salary_data = r"C:\Users\patri\Downloads\salaryData.csv"
salary_df = pd.read_csv(salary_data) # Conversion of the csv file into a data frame using pandas.
print(salary_df.head(15)) # Some data inspection of the first 15 rows of the data. 
print(salary_df.isnull().sum()) # Explanatory data analysis to determine how many empty columns cells are.