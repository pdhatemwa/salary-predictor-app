import streamlit as st 
st.title("🎈 My new app")
st.write(
     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

first_name = st.text_input("First name")
last_name = st.text_input("Last name")
gender = st.number_input("Your age", 0, 100, 30, 1)
dob = st.date_input("Your birthday")
marital_status = st.radio("Marital status", ["Single", "Married"])
years_of_experience = st.slider("Years of experience", 0, 40)
