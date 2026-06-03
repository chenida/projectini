import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Klasifikasi Transaksi", layout="wide")

# Title dashboard
st.title("📊 Dashboard Klasifikasi Kategori Transaksi")
st.markdown("Visualisasi perbandingan data **Training** dan **Testing** tanpa nilai rupiah")

# Load data dari file yang diupload atau dari file yang sudah ada
uploaded_file = st.file_uploader("Upload file CSV (dataset_transaksi.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # Gunakan data dari file yang disediakan (simulasikan sebagai string)
    st.info("Menggunakan dataset default. Upload file CSV jika ingin mengganti.")
    # Simulasikan data dari file content yang diberikan
    data_content = """deskripsi,kategori
profit trading crypto,Investasi
denda telat bayar,Lainnya
fotocopy berkas,Lainnya
skincare soco,Belanja
uang kas rt,Lainnya
jual sbn,Investasi
profit trading crypto,Investasi
pencairan reksadana,Investasi
isi bensin pertamax,Transportasi
reward,Bonus
... (data lengkap dari file, potong untuk contoh)"""
    # Karena data sangat panjang, kita akan membaca dari file yang diupload saja
    st.warning("Silakan upload file dataset_transaksi.csv untuk melanjutkan.")
    st.stop()

# Pastikan kolom yang diperlukan ada
if 'deskripsi' not in df.columns or 'kategori' not in df.columns:
    st.error("File harus memiliki kolom 'deskripsi' dan 'kategori'")
    st.stop()

# Drop missing values
df = df.dropna(subset=['deskripsi', 'kategori'])

# Tampilkan info dataset
st.subheader("📋 Informasi Dataset")
col1, col2, col3 = st.columns(3)
col1.metric("Total Transaksi", len(df))
col2.metric("Unique Kategori", df['kategori'].nunique())
col3.metric("Unique Deskripsi", df['deskripsi'].nunique())

# Distribusi kategori sebelum split
st.subheader("📈 Distribusi Kategori (Sebelum Split)")
fig, ax = plt.subplots(figsize=(10, 5))
df['kategori'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
ax.set_title("Frekuensi Kategori - Seluruh Data")
ax.set_xlabel("Kategori")
ax.set_ylabel("Frekuensi")
plt.xticks(rotation=45)
st.pyplot(fig)

# Split data menjadi training dan testing (70:30)
X = df['deskripsi']
y = df['kategori']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Tampilkan ukuran training dan testing
st.subheader("🔀 Pembagian Data Training dan Testing")
col1, col2 = st.columns(2)
col1.metric("Training Set", f"{len(X_train)} transaksi ({len(X_train)/len(df)*100:.1f}%)")
col2.metric("Testing Set", f"{len(X_test)} transaksi ({len(X_test)/len(df)*100:.1f}%)")

# Fitur: TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model klasifikasi
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# Akurasi
accuracy = accuracy_score(y_test, y_pred)
st.metric("Akurasi Model (Naive Bayes)", f"{accuracy*100:.2f}%")

# Tampilkan perbandingan distribusi kategori antara training dan testing
st.subheader("📊 Perbandingan Distribusi Kategori: Training vs Testing")

train_dist = y_train.value_counts()
test_dist = y_test.value_counts()
compare_df = pd.DataFrame({
    'Training': train_dist,
    'Testing': test_dist
}).fillna(0).astype(int)

fig, ax = plt.subplots(figsize=(12, 6))
compare_df.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
ax.set_title("Perbandingan Frekuensi Kategori pada Training dan Testing")
ax.set_xlabel("Kategori")
ax.set_ylabel("Frekuensi")
plt.xticks(rotation=45)
st.pyplot(fig)

# Tabel perbandingan
st.subheader("📋 Tabel Perbandingan Training vs Testing")
st.dataframe(compare_df, use_container_width=True)

# Confusion Matrix
st.subheader("🧩 Confusion Matrix (Testing Set)")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# Classification Report
st.subheader("📄 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.round(4), use_container_width=True)

# Visualisasi perbandingan proporsi
st.subheader("🥧 Perbandingan Proporsi Kategori (Pie Chart)")
col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    train_dist.plot(kind='pie', autopct='%1.1f%%', ax=ax1, startangle=90)
    ax1.set_title("Training Set")
    ax1.set_ylabel("")
    st.pyplot(fig1)
with col2:
    fig2, ax2 = plt.subplots()
    test_dist.plot(kind='pie', autopct='%1.1f%%', ax=ax2, startangle=90)
    ax2.set_title("Testing Set")
    ax2.set_ylabel("")
    st.pyplot(fig2)

# Contoh prediksi manual
st.subheader("🔍 Coba Prediksi Manual")
user_input = st.text_input("Masukkan deskripsi transaksi:", "beli bensin shell")
if user_input:
    user_vec = vectorizer.transform([user_input])
    pred = model.predict(user_vec)[0]
    proba = model.predict_proba(user_vec)[0]
    st.success(f"Prediksi Kategori: **{pred}**")
    # Tampilkan probabilitas
    prob_df = pd.DataFrame({
        'Kategori': model.classes_,
        'Probabilitas': proba
    }).sort_values('Probabilitas', ascending=False)
    st.bar_chart(prob_df.set_index('Kategori'))

# Footer
st.markdown("---")
st.caption("Dashboard ini hanya menampilkan data kategorikal tanpa nilai rupiah. Model: Naive Bayes dengan TF-IDF.")