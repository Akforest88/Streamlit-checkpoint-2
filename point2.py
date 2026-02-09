import streamlit as st
import pandas as pd
import joblib

# Charger le modèle
model = joblib.load("financial_inclusion_model.pkl")

st.title("💰 Prédiction d’Inclusion Financière")

st.write("Veuillez entrer les informations du répondant")

# Champs de saisie (exemples)
country = st.number_input("Pays", min_value=0)
year = st.number_input("Année", min_value=2000, max_value=2030)
age = st.number_input("Âge", min_value=10, max_value=100)
gender = st.selectbox("Genre", [0, 1])
education = st.number_input("Niveau d’éducation", min_value=0)
job_type = st.number_input("Type d’emploi", min_value=0)

# Bouton
if st.button("Prédire"):
    input_data = pd.DataFrame([[country, year, age, gender, education, job_type]],
                              columns=[
                                  "country", "year", "age_of_respondent",
                                  "gender_of_respondent",
                                  "education_level",
                                  "job_type"
                              ])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Le répondant possède un compte bancaire")
    else:
        st.warning("❌ Le répondant ne possède pas de compte bancaire")