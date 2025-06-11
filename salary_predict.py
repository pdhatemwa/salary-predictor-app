import streamlit as st 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os
import datetime
from datetime import date


#os is a built-in Python module that helps you interact with the Operating System — 
#like dealing with files, directories, paths, etc.
#We use os here to make our file path dynamic and portable.


st.title("Wage prediction app") # App title

first_name = st.text_input("First name") # Ask user to input first name. "text_input" for text.
last_name = st.text_input("Last name") # Ask user to input first name. "text_input" for text.

today = date.today() # Setting today's date to accurately calculate the age.
dob = st.date_input(
     "Select your date of birth",
     value = datetime.date(1970, 1, 1),  # default year value to cater to older users if any.
     min_value = datetime.date(1900, 1, 1),
     max_value = datetime.date.today()
)
if dob > today: # To calculate the user's age based on date of birth.
     st.error("Date of birth cannot be in the future!") # Error handling for notorious users.
else:
     age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
     st.success(f"Your age is: {age} years")

marital_status = st.radio("Marital status", ["Single", "Married"]) # To show marital status even though it isn't a strong wage predictor.

education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"]) # User to select education level.

st.write("Use the slider to indicate how many years of experience you have.") # Some user instructions.

years_of_experience = st.slider("Years of experience", 0, 40) # since the average career lasts up to 40 years.

job_title = st.selectbox("Job Title", ["Software Engineer", "Data Analyst", "Manager", "Teacher", 
                                   "Senior Scientist", "Financial Analyst", "Accountant",
                                   "CEO", "Business Analyst"])

# salary_data = r"C:\Users\patri\Downloads\salaryData.csv" as the absolute file path to the csv data
# But cloud based apps don't read absolute file paths but rather relative paths which in this case would just be "salaryData.csv"

salary_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "salaryData.csv"))

# Conversion of the csv file into a data frame using pandas.
# print(salary_df.head(15)) # Some data inspection of the first 15 rows of the data. 
# print(salary_df.isnull().sum()) # Explanatory data analysis to determine how many empty columns cells are.
# print(salary_df.describe)

# Setting up X and Y values for model prediction.
Y = salary_df['Salary']
feature_names = ['Age', 'Education Level', 'Job Title', 'Years of Experience']
X = salary_df[feature_names]

# Combining features X and Y side by side into a single data frame.
# dropna() drops cells with empty cells down the columns.
combined = pd.concat([X, Y], axis = 1).dropna()

X_clean = combined[feature_names]
Y_clean = combined['Salary']

# Applying one-hot encode for categorical variables.
X_encoded = pd.get_dummies(X_clean)

# Split the data into training data and test data.
# By default, train_test_split() splits 75% for training and 25% for validation.
# By setting random_state = 1, or any number, it is shuffling the same way every time so I can get consistent results.
X_train, X_valid, Y_train, Y_valid = train_test_split(X_encoded, Y_clean, random_state = 1)

# Prepare user input row for prediction
# Here I am basically creating rows for user input into their own dataset.
# The purpose of this is to bundle the user's input into a dictionary so it can be converted into a DataFrame later.
input_dict = {
     'Education Level': [education],
     'Job Title': [job_title],
     'Years of Experience': [years_of_experience],
     'Age' : [age]
}
user_df = pd.DataFrame(input_dict) # This turns the dictionary into a data frame.



# Fitting the model using random forest regressor a fairly accurate M/L algorithm.
model = RandomForestRegressor(random_state = 1)
model.fit(X_train, Y_train)
user_df = pd.DataFrame(input_dict)

# One-hot encode user input converts categorical variables like Job Title and Education Level into numerical values.
user_encoded = pd.get_dummies(user_df)

# This is to make sure that the user input matches exactly the same columns as the training data.
for col in X_encoded.columns:
     if col not in user_encoded:
          user_encoded[col] = 0

# Reorder columns after filling them in in the previous code block.
user_encoded = user_encoded[X_encoded.columns]

# Make predictions based on the data.
predictions = model.predict(X_valid)
mae = mean_absolute_error(Y_valid, predictions) # To test just how accurate the model is.
# print("\n")
# print(f"Mean absolute Error : {mae}")
# percentage_mae = (mae/Y_clean.mean())*100
# print(f"Percentage MAE : {percentage_mae}") # %mean error = 9.1946

# Salary prediction button.
if st.button("Predict Salary"):
     predicted_salary = model.predict(user_encoded)[0]
     st.success(f"Estimated Salary of {first_name} : ${predicted_salary:,.2f} per year.")

