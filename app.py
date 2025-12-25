import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# Judul Aplikasi
st.title("🎓 Aplikasi Prediksi Kelulusan Mahasiswa")
st.markdown("Menggunakan Algoritma Gaussian Naive Bayes")
st.write("---")

# ==========================================
# 2. LOAD DATA & TRAIN MODEL (CACHED)
# ==========================================
@st.cache_data
def load_and_train_model():
    # Load Data
    try:
        df = pd.read_csv('dataset (1).csv')
    except FileNotFoundError:
        st.error("File 'dataset (1).csv' tidak ditemukan! Pastikan file ada di folder yang sama.")
        return None, None, None, None, None

    # Select Features
    df_model = df[['Gender', 'Age at enrollment', 'Curricular units 2nd sem (grade)', 
                   'Debtor', 'Tuition fees up to date', 'Target']].copy()
    df_model.columns = ['Gender', 'Age', 'Grade', 'Debtor', 'Tuition_Paid', 'Target']

    # Filter & Encode Target
    df_model = df_model[df_model['Target'] != 'Enrolled'].copy()
    df_model['Target'] = df_model['Target'].map({'Dropout': 0, 'Graduate': 1})

    # Split X & y
    X = df_model.drop('Target', axis=1)
    y = df_model['Target']

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train Model
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Return everything needed
    return model, scaler, X_test, y_test, df_model

# Eksekusi fungsi training
model, scaler, X_test, y_test, df_full = load_and_train_model()

# Jika data gagal dimuat, hentikan aplikasi
if model is None:
    st.stop()

# ==========================================
# 3. SIDEBAR: FORM INPUT USER
# ==========================================
st.sidebar.header("📝 Input Data Mahasiswa")
st.sidebar.write("Masukkan data mahasiswa baru untuk diprediksi:")

# Input Form
gender_input = st.sidebar.selectbox("Gender", ["Male", "Female"])
age_input = st.sidebar.slider("Umur (Age)", 17, 60, 20)
grade_input = st.sidebar.slider("Nilai Semester 2 (Grade)", 0.0, 20.0, 14.0)
debtor_input = st.sidebar.radio("Apakah Punya Hutang?", ["Tidak", "Ya"])
tuition_input = st.sidebar.radio("Status Bayar SPP?", ["Lancar", "Menunggak"])

# Konversi Input ke Angka (Sesuai Preprocessing)
gender_val = 1 if gender_input == "Male" else 0
debtor_val = 1 if debtor_input == "Ya" else 0
tuition_val = 1 if tuition_input == "Lancar" else 0

# Tombol Prediksi
predict_btn = st.sidebar.button("Prediksi Sekarang 🚀")

# ==========================================
# 4. MAIN AREA: HASIL PREDIKSI
# ==========================================
if predict_btn:
    st.subheader("🔍 Hasil Prediksi")
    
    # Siapkan data baru dan lakukan SCALING
    input_data = np.array([[gender_val, age_input, grade_input, debtor_val, tuition_val]])
    input_scaled = scaler.transform(input_data)
    
    # Prediksi
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Tampilan Hasil
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.success("✅ Prediksi: GRADUATE (Lulus)")
            st.image("https://media.giphy.com/media/3o6fJ1BM7R2EBRDnxK/giphy.gif", caption="Selamat!", width=300)
        else:
            st.error("⚠️ Prediksi: DROPOUT")
            st.image("https://media.giphy.com/media/l2Je3qSgV5O4X8Y5G/giphy.gif", caption="Perlu Perhatian Lebih", width=300)
            
    with col2:
        st.write("Tingkat Keyakinan Model:")
        st.progress(int(probability[prediction] * 100))
        st.write(f"Probabilitas: {probability[prediction]*100:.2f}%")
        
        st.info(f"""
        Detail Input:
        - Umur: {age_input} Tahun
        - Nilai: {grade_input}
        - Hutang: {debtor_input}
        - SPP: {tuition_input}
        """)

st.write("---")

# ==========================================
# 5. VISUALISASI DATA & MODEL
# ==========================================
st.subheader("📊 Visualisasi Performa Model")

tab1, tab2, tab3 = st.tabs(["Distribusi Data (KDE)", "Evaluasi (Confusion Matrix)", "3D Plot (Interaktif)"])

with tab1:
    st.write("Perbandingan Nilai & Umur: Graduate vs Dropout")
    fig_kde, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Grade
    sns.kdeplot(df_full.loc[df_full['Target'] == 0, 'Grade'], label='Dropout', fill=True, color='red', ax=ax[0])
    sns.kdeplot(df_full.loc[df_full['Target'] == 1, 'Grade'], label='Graduate', fill=True, color='green', ax=ax[0])
    ax[0].set_title('Distribusi Nilai')
    ax[0].legend()
    
    # Plot Age
    sns.kdeplot(df_full.loc[df_full['Target'] == 0, 'Age'], label='Dropout', fill=True, color='red', ax=ax[1])
    sns.kdeplot(df_full.loc[df_full['Target'] == 1, 'Age'], label='Graduate', fill=True, color='green', ax=ax[1])
    ax[1].set_title('Distribusi Umur')
    ax[1].set_xlim(17, 50)
    ax[1].legend()
    
    st.pyplot(fig_kde)

with tab2:
    st.write("Confusion Matrix")
    y_pred_test = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    acc = accuracy_score(y_test, y_pred_test)
    
    st.metric("Akurasi Model", f"{acc*100:.2f}%")
    
    fig_cm = plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Dropout', 'Graduate'], 
                yticklabels=['Dropout', 'Graduate'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig_cm)

with tab3:
    st.write("Visualisasi 3D (Putar grafik untuk melihat detail)")
    
    # Reconstruct data test untuk visualisasi
    X_test_inv = scaler.inverse_transform(X_test)
    df_viz = pd.DataFrame(X_test_inv, columns=['Gender', 'Age', 'Grade', 'Debtor', 'Tuition_Paid'])
    df_viz['Prediksi'] = y_pred_test
    df_viz['Prediksi'] = df_viz['Prediksi'].map({0: 'Dropout', 1: 'Graduate'})
    df_viz['Status_Asli'] = y_test.values
    df_viz['Status_Asli'] = df_viz['Status_Asli'].map({0: 'Dropout', 1: 'Graduate'})
    
    fig_3d = px.scatter_3d(
        data_frame=df_viz,
        x='Age',
        y='Grade',
        z='Status_Asli',
        color='Prediksi',
        color_discrete_map={'Dropout': 'red', 'Graduate': 'green'},
        opacity=0.6,
        height=600,
        title='Umur vs Nilai vs Status'
    )
    st.plotly_chart(fig_3d, use_container_width=True)