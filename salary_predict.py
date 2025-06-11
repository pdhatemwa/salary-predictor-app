import streamlit as st 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sns
st.title("Wage prediction app")
st.write("ML app")

first_name = st.text_input("First name")
last_name = st.text_input("Last name")
gender = st.number_input("Your age", 0, 100, 30, 1)
marital_status = st.radio("Marital status", ["Single", "Married"])
education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
years_of_experience = st.slider("Years of experience", 0, 40)
job_title = st.selectbox("Job Title", ["Software Engineer", "Data Analyst", "Manager", "Teacher"])

# salary_data = r"C:\Users\patri\Downloads\salaryData.csv"
salary_df = pd.read_csv("salaryData.csv") # Conversion of the csv file into a data frame using pandas.
# print(salary_df.head(15)) # Some data inspection of the first 15 rows of the data. 
# print(salary_df.isnull().sum()) # Explanatory data analysis to determine how many empty columns cells are.
# print(salary_df.describe)

# Setting up X and Y values for model prediction.
Y = salary_df['Salary']
feature_names = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
X = salary_df[feature_names]

# Combining features X and Y side by side into a single data frame.
combined = pd.concat([X, Y], axis = 1).dropna()
X_clean = combined[feature_names]
Y_clean = combined['Salary']

# Applying one-hot encode for categorical variables.
X_encoded = pd.get_dummies(X_clean)

# Split the data into training data and test data.
X_train, X_valid, Y_train, Y_valid = train_test_split(X_encoded, Y_clean, random_state = 1)

# 3. Prepare user input row for prediction
input_dict = {
     'Education Level': [education],
     'Job Title': [job_title],
     'Years of Experience': [years_of_experience]
}
user_df = pd.DataFrame(input_dict)
# Model fitting
model = RandomForestRegressor(random_state = 1)
model.fit(X_train, Y_train)
user_df = pd.DataFrame(input_dict)

# 4. One-hot encode user input to match training features
user_encoded = pd.get_dummies(user_df)

# Ensure same columns as training set
for col in X_encoded.columns:
     if col not in user_encoded:
          user_encoded[col] = 0

# Reorder columns
user_encoded = user_encoded[X_encoded.columns]

# 5. Make prediction
predictions = model.predict(X_valid)
mae = mean_absolute_error(Y_valid, predictions)
# print("\n")
# print(f"Mean absolute Error : {mae}")
# percentage_mae = (mae/Y_clean.mean())*100
# print(f"Percentage MAE : {percentage_mae}") # %mean error = 9.1946
if st.button("Predict Salary"):
     predicted_salary = model.predict(user_encoded)[0]
     st.success(f"Estimated Salary of {first_name} : ${predicted_salary:,.2f} per year.")

