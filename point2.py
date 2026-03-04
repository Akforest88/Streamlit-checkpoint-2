import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Inclusion Financière Afrique", page_icon="💰", layout="centered")

def local_css():
    st.markdown("""
        <style>
        .main { background-color: #f8f9fa; }
        .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
        .result-card { padding: 20px; border-radius: 10px; border: 1px solid #ddd; background-color: white; }
        </style>
        """, unsafe_allow_html=True)

local_css()

# --- 2. CONSTANTES & MAPPINGS ---
# Centraliser les données facilite la mise à jour sans toucher à la logique de calcul
MAPPINGS = {
    "country": ["Burundi", "Djibouti", "Érythrée", "Éthiopie", "Kenya", "Ouganda", "Rwanda", "Somalie", "Soudan du Sud", "Tanzanie"],
    "education": ["Aucun", "Primaire", "Secondaire", "Supérieur"],
    "job": ["Sans emploi", "Employé", "Indépendant", "Agriculteur", "Autre"],
    "gender": {"Homme": 1, "Femme": 0}
}

# --- 3. LOGIQUE MÉTIER ---
@st.cache_resource
def load_model(path="financial_inclusion_model.pkl"):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

model = load_model()

# --- 4. INTERFACE ---
st.title("💰 Prédiction d'Inclusion Financière")
st.markdown("Estimez la probabilité d'accès aux services bancaires en Afrique de l'Est.")

if model is None:
    st.error("❌ **Fichier modèle introuvable.** Veuillez placer `financial_inclusion_model.pkl` à la racine.")
    st.stop()

# Utilisation de colonnes pour un formulaire plus compact
with st.container():
    st.subheader("📋 Profil de l'individu")
    col1, col2 = st.columns(2)
    
    with col1:
        country = st.selectbox("🌍 Pays", MAPPINGS["country"])
        age = st.slider("🎂 Âge", 18, 100, 30)
        gender = st.radio("👤 Genre", list(MAPPINGS["gender"].keys()), horizontal=True)
    
    with col2:
        education = st.selectbox("🎓 Niveau d'études", MAPPINGS["education"])
        job = st.selectbox("💼 Type d'emploi", MAPPINGS["job"])
        year = st.number_input("📅 Année de référence", 2010, 2030, 2024)

    predict_btn = st.button("🔍 Analyser le profil")

# --- 5. TRAITEMENT & RÉSULTATS ---
if predict_btn:
    try:
        # Encodage propre via les index des listes de mapping
        input_df = pd.DataFrame([{
            "country": MAPPINGS["country"].index(country),
            "year": year,
            "age_of_respondent": age,
            "gender_of_respondent": MAPPINGS["gender"][gender],
            "education_level": MAPPINGS["education"].index(education),
            "job_type": MAPPINGS["job"].index(job)
        }])

        # Inférence avec spinner pour le feedback visuel
        with st.spinner('Analyse en cours...'):
            prob = model.predict_proba(input_df)[0][1]
            is_banked = model.predict(input_df)[0]

        st.markdown("---")
        
        # Affichage stylisé
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            if is_banked == 1:
                st.success("### ✅ Accès probable")
                st.write("L'individu correspond au profil type des utilisateurs bancarisés.")
            else:
                st.warning("### ❌ Accès limité")
                st.write("L'individu pourrait rencontrer des freins à l'inclusion financière.")
        
        with res_col2:
            st.metric("Indice de confiance", f"{prob:.1%}")
            st.progress(prob)

        # Expander pour plus de détails techniques
        with st.expander("🔬 Détails techniques"):
            st.json(input_df.to_dict(orient='records')[0])
            st.write("Le modèle utilise un algorithme de classification pour pondérer les facteurs socio-économiques.")

    except Exception as e:
        st.error(f"Une erreur est survenue lors de la prédiction : {e}")
