import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Klasifikasi Transaksi - LSTM", layout="wide")

# Title dashboard
st.title("🧠 Dashboard Klasifikasi Kategori Transaksi dengan LSTM")
st.markdown("Visualisasi perbandingan data **Training** dan **Testing** menggunakan Deep Learning (LSTM)")

# Load data dari file yang diupload
uploaded_file = st.file_uploader("Upload file CSV (dataset_transaksi.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
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

# Split data menjadi training dan testing (80:20 untuk deep learning)
X = df['deskripsi'].values
y = df['kategori'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tampilkan ukuran training dan testing
st.subheader("🔀 Pembagian Data Training dan Testing")
col1, col2 = st.columns(2)
col1.metric("Training Set", f"{len(X_train)} transaksi ({len(X_train)/len(df)*100:.1f}%)")
col2.metric("Testing Set", f"{len(X_test)} transaksi ({len(X_test)/len(df)*100:.1f}%)")

# Parameter LSTM
MAX_FEATURES = 5000  # Vocabulary size
MAX_LEN = 50  # Maximum sequence length
EMBEDDING_DIM = 100  # Embedding dimension
LSTM_UNITS = 64
BATCH_SIZE = 32
EPOCHS = 20

# Preprocessing untuk LSTM
st.subheader("⚙️ Preprocessing Data untuk LSTM")
with st.spinner("Melakukan tokenisasi dan padding..."):
    # Tokenizer
    tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    # Convert to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Padding
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Label Encoding
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    
    # One-hot encoding untuk multiclass classification
    num_classes = len(label_encoder.classes_)
    y_train_cat = tf.keras.utils.to_categorical(y_train_enc, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test_enc, num_classes)

st.success(f"Preprocessing selesai! Vocabulary size: {len(tokenizer.word_index)} kelas, Max sequence length: {MAX_LEN}")

# Tampilkan contoh hasil preprocessing
with st.expander("Lihat contoh hasil preprocessing"):
    st.write("**Contoh deskripsi asli:**")
    st.write(X_train[:3])
    st.write("**Contoh sequence setelah tokenisasi:**")
    st.write(X_train_seq[:3])
    st.write("**Contoh setelah padding:**")
    st.write(X_train_pad[:3])

# Build LSTM Model
st.subheader("🏗️ Arsitektur Model LSTM")
with st.expander("Lihat detail arsitektur model"):
    st.code("""
    Model: Sequential
    ├─ Embedding(vocab_size=5000, output_dim=100, input_length=50)
    ├─ Bidirectional(LSTM(64, return_sequences=True))
    ├─ Dropout(0.3)
    ├─ LSTM(32)
    ├─ Dropout(0.3)
    ├─ Dense(64, activation='relu')
    ├─ Dropout(0.3)
    └─ Dense(num_classes, activation='softmax')
    """)

# Build model
model = Sequential([
    Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Tampilkan summary model
st.text("Model Summary:")
summary_str = []
model.summary(print_fn=lambda x: summary_str.append(x))
st.code('\n'.join(summary_str))

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Train model dengan progress bar
st.subheader("🎯 Training Model LSTM")
progress_bar = st.progress(0)
status_text = st.empty()

# Custom callback untuk update progress
class StreamlitProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, status_text):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.current_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1
        progress = self.current_epoch / EPOCHS
        self.progress_bar.progress(progress)
        self.status_text.text(f"Epoch {self.current_epoch}/{EPOCHS} - Loss: {logs['loss']:.4f} - Acc: {logs['accuracy']:.4f} - Val Loss: {logs['val_loss']:.4f} - Val Acc: {logs['val_accuracy']:.4f}")

# Training
with st.spinner("Melakukan training model LSTM (ini mungkin memakan waktu beberapa menit)..."):
    history = model.fit(
        X_train_pad, y_train_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping, StreamlitProgressCallback(progress_bar, status_text)],
        verbose=0
    )

# Evaluasi model
st.subheader("📊 Evaluasi Model")
y_pred_prob = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

accuracy = accuracy_score(y_true, y_pred)
st.metric("Akurasi Model LSTM", f"{accuracy*100:.2f}%")

# Learning Curve
st.subheader("📈 Learning Curve")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss plot
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

st.pyplot(fig)

# Tampilkan perbandingan distribusi kategori antara training dan testing
st.subheader("📊 Perbandingan Distribusi Kategori: Training vs Testing")

train_dist = pd.Series(y_train).value_counts()
test_dist = pd.Series(y_test).value_counts()
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

# Confusion Matrix dengan Plotly interaktif
st.subheader("🧩 Confusion Matrix (Testing Set)")
cm = confusion_matrix(y_true, y_pred)
class_names = label_encoder.classes_

fig_cm = px.imshow(
    cm,
    x=class_names,
    y=class_names,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='Blues',
    title="Confusion Matrix - LSTM"
)
fig_cm.update_layout(
    xaxis_title="Predicted",
    yaxis_title="Actual",
    width=800,
    height=600
)
st.plotly_chart(fig_cm)

# Classification Report
st.subheader("📄 Classification Report")
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.round(4), use_container_width=True)

# Visualisasi perbandingan proporsi
st.subheader("🥧 Perbandingan Proporsi Kategori")
col1, col2 = st.columns(2)
with col1:
    fig1 = px.pie(
        values=train_dist.values,
        names=train_dist.index,
        title="Training Set Distribution",
        hole=0.3
    )
    st.plotly_chart(fig1, use_container_width=True)
    
with col2:
    fig2 = px.pie(
        values=test_dist.values,
        names=test_dist.index,
        title="Testing Set Distribution",
        hole=0.3
    )
    st.plotly_chart(fig2, use_container_width=True)

# Prediksi manual
st.subheader("🔍 Coba Prediksi Manual dengan LSTM")
user_input = st.text_input("Masukkan deskripsi transaksi:", "beli bensin shell")
if user_input:
    # Preprocess input
    user_seq = tokenizer.texts_to_sequences([user_input])
    user_pad = pad_sequences(user_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Predict
    pred_prob = model.predict(user_pad, verbose=0)
    pred_class = np.argmax(pred_prob, axis=1)[0]
    pred_label = label_encoder.inverse_transform([pred_class])[0]
    
    st.success(f"Prediksi Kategori: **{pred_label}**")
    
    # Tampilkan probabilitas semua kelas
    prob_df = pd.DataFrame({
        'Kategori': class_names,
        'Probabilitas': pred_prob[0] * 100
    }).sort_values('Probabilitas', ascending=False)
    
    # Bar chart dengan Plotly
    fig_prob = px.bar(
        prob_df,
        x='Kategori',
        y='Probabilitas',
        title='Probabilitas Prediksi per Kategori',
        color='Probabilitas',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_prob, use_container_width=True)

# Simpan model (opsional)
st.subheader("💾 Simpan Model")
if st.button("Simpan Model LSTM"):
    model.save('lstm_transaction_classifier.h5')
    st.success("Model berhasil disimpan sebagai 'lstm_transaction_classifier.h5'")

# Footer
st.markdown("---")
st.caption("Dashboard ini menggunakan model LSTM (Deep Learning) dengan arsitektur Bidirectional LSTM. Akurasi mungkin lebih baik dibandingkan Naive Bayes untuk data teks yang kompleks.")