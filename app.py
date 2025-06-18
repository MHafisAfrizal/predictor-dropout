import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page config (di awal setelah import st)
st.set_page_config(page_title='Dropout Probability Predictor', layout='wide')

# Load model
@st.cache_resource
def load_model(path='./model/model.pkl'):
    return joblib.load(path)

model = load_model()

st.title('📉 Dropout Probability Predictor')

# Input untuk prediksi
st.subheader('Masukkan Data Siswa')
age = st.number_input('Usia pada Pendaftaran', min_value=17, max_value=70, value=20)
units_approved = st.number_input('Unit Disetujui Semester 1', min_value=0, max_value=20, value=5)

# Buat input_data dengan semua fitur yang mungkin
if st.button('Hitung Probabilitas Dropout'):
    input_data = pd.DataFrame({
        'Marital_status': [0], 'Application_mode': [0], 'Application_order': [0], 'Course': [0],
        'Daytime_evening_attendance': [0], 'Previous_qualification': [0], 'Previous_qualification_grade': [0],
        'Nacionality': [0], 'Mothers_qualification': [0], 'Fathers_qualification': [0],
        'Mothers_occupation': [0], 'Fathers_occupation': [0], 'Admission_grade': [0], 'Displaced': [0],
        'Educational_special_needs': [0], 'Debtor': [0], 'Tuition_fees_up_to_date': [0], 'Gender': [0],
        'Scholarship_holder': [0], 'Age_at_enrollment': [age], 'International': [0],
        'Curricular_units_1st_sem_credited': [0], 'Curricular_units_1st_sem_enrolled': [0],
        'Curricular_units_1st_sem_evaluations': [0], 'Curricular_units_1st_sem_approved': [units_approved],
        'Curricular_units_1st_sem_grade': [0], 'Curricular_units_1st_sem_without_evaluations': [0],
        'Curricular_units_2nd_sem_credited': [0], 'Curricular_units_2nd_sem_enrolled': [0],
        'Curricular_units_2nd_sem_evaluations': [0], 'Curricular_units_2nd_sem_approved': [0],
        'Curricular_units_2nd_sem_grade': [0], 'Curricular_units_2nd_sem_without_evaluations': [0],
        'Unemployment_rate': [0], 'Inflation_rate': [0], 'GDP': [0]
    })
    # Hitung probabilitas
    try:
        prob = model.predict_proba(input_data)
        dropout_prob = prob[0][0] * 100  # Asumsi indeks 0 adalah probabilitas Dropout
        color = 'red' if dropout_prob > 50 else 'green'
        st.markdown(f"<h3 style='color:{color};'>Probabilitas Dropout: {dropout_prob:.2f}%</h3>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {e}. Pastikan model support predict_proba atau cek fitur.")

# Catatan
st.caption('Aplikasi ini menghitung risiko dropout berdasarkan model terlatih. Visualisasi ada di Metabase.')