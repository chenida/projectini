# dashboard_keuangan.py (VERSI DIPERBAIKI)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Set page config
st.set_page_config(
    page_title="Dashboard Keuangan - Klasifikasi & Prediksi",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E5A88;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E3A5F;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    # Create comprehensive dataset
    np.random.seed(42)
    
    categories = ['Investasi', 'Belanja', 'Transportasi', 'Makanan & Minuman', 
                  'Hiburan', 'Tagihan', 'Bonus', 'Gaji', 'Lainnya']
    
    # Common transaction descriptions
    descriptions = {
        'Investasi': ['profit trading crypto', 'jual sbn', 'pencairan reksadana', 'bunga deposito', 'dividen saham'],
        'Belanja': ['skincare soco', 'sabun mandi', 'belanja bulanan indomaret', 'beli jaket', 'beli sepatu tokopedia', 'baju di shopee'],
        'Transportasi': ['isi bensin pertamax', 'bayar parkir mall', 'beli tiket kereta', 'tambal ban', 'gojek ke stasiun', 'grab car kantor', 'topup flazz mrt'],
        'Makanan & Minuman': ['makan siang warteg', 'gofood sate', 'kopi janji jiwa', 'nongkrong cafe', 'grabfood mcd', 'beli nasi padang', 'beli cemilan minimarket', 'beli air galon'],
        'Hiburan': ['langganan spotify', 'beli game steam', 'nonton bioskop xxi', 'main timezone', 'beli tiket konser', 'top up diamond mobile legends'],
        'Tagihan': ['bayar listrik pln', 'tagihan air pdam', 'langganan netflix', 'cicilan motor', 'bayar bpjs', 'beli token listrik', 'tagihan indihome'],
        'Bonus': ['reward', 'insentif pencapaian', 'bonus akhir tahun', 'bonus proyek', 'thr', 'cashback'],
        'Gaji': ['gaji bulanan mei', 'transfer gaji pt jan', 'penerimaan gaji apr', 'gaji masuk jan', 'payroll bulan mei', 'gaji bulanan apr'],
        'Lainnya': ['denda telat bayar', 'fotocopy berkas', 'uang kas rt', 'biaya admin bank', 'sumbangan mesjid', 'kasih pengemis']
    }
    
    # Generate 2000 rows
    n_samples = 2000
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', periods=n_samples)
    
    amounts = []
    generated_descriptions = []
    generated_categories = []
    
    for i, date in enumerate(dates):
        month = date.month
        day = date.day
        is_early_year = month in [1, 2]
        
        # Choose category with probability
        if np.random.random() < 0.2:
            category = np.random.choice(categories)
        else:
            weights = [0.15, 0.15, 0.12, 0.12, 0.10, 0.10, 0.10, 0.08, 0.08]
            category = np.random.choice(categories, p=weights)
        
        desc = np.random.choice(descriptions[category])
        
        # Generate amount based on category
        if category == 'Gaji':
            base_amount = np.random.uniform(4000000, 8000000)
            if is_early_year:
                base_amount *= np.random.uniform(1.05, 1.15)
        elif category == 'Bonus':
            base_amount = np.random.uniform(500000, 3000000)
            if is_early_year:
                base_amount *= np.random.uniform(1.1, 1.3)
        elif category == 'Investasi':
            base_amount = np.random.uniform(100000, 2000000)
        elif category == 'Belanja':
            base_amount = np.random.uniform(50000, 500000)
            if day > 25:
                base_amount *= np.random.uniform(1.2, 1.5)
        elif category == 'Makanan & Minuman':
            base_amount = np.random.uniform(15000, 150000)
        elif category == 'Transportasi':
            base_amount = np.random.uniform(10000, 200000)
        elif category == 'Hiburan':
            base_amount = np.random.uniform(25000, 300000)
        elif category == 'Tagihan':
            base_amount = np.random.uniform(50000, 500000)
        else:
            base_amount = np.random.uniform(10000, 100000)
        
        amount = int(base_amount * np.random.uniform(0.8, 1.2))
        
        amounts.append(amount)
        generated_descriptions.append(desc)
        generated_categories.append(category)
    
    df_full = pd.DataFrame({
        'tanggal': dates,
        'deskripsi': generated_descriptions,
        'kategori': generated_categories,
        'jumlah': amounts
    })
    
    # Add derived columns
    df_full['tahun'] = df_full['tanggal'].dt.year
    df_full['bulan'] = df_full['tanggal'].dt.month
    df_full['bulan_nama'] = df_full['tanggal'].dt.strftime('%B')
    df_full['hari'] = df_full['tanggal'].dt.day
    df_full['minggu_ke'] = ((df_full['hari'] - 1) // 7) + 1
    df_full['is_early_year'] = df_full['bulan'].isin([1, 2]).astype(int)
    df_full['type'] = df_full['kategori'].apply(
        lambda x: 'Pemasukan' if x in ['Gaji', 'Bonus'] 
        else ('Investasi' if x == 'Investasi' else 'Pengeluaran')
    )
    
    return df_full

@st.cache_resource
def train_classification_model(df):
    """Train Random Forest for transaction classification"""
    # TF-IDF on description
    tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    X_text = tfidf.fit_transform(df['deskripsi']).toarray()
    
    # Add amount feature
    X_amount = df[['jumlah']].values
    
    # Combine features
    X = np.hstack([X_text, X_amount])
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df['kategori'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, tfidf, le, accuracy

@st.cache_resource
def train_timeseries_model(df):
    """Train LSTM/GRU model for monthly prediction"""
    # Aggregate by month
    monthly = df.copy()
    
    # Calculate monthly totals using pivot_table (FIXED)
    monthly_pivot = monthly.pivot_table(
        index=['tahun', 'bulan', 'is_early_year'],
        columns='type',
        values='jumlah',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    # Rename columns
    monthly_pivot = monthly_pivot.rename(columns={
        'Pemasukan': 'pemasukan',
        'Pengeluaran': 'pengeluaran',
        'Investasi': 'investasi'
    })
    
    # If Investasi column doesn't exist, add it
    if 'investasi' not in monthly_pivot.columns:
        monthly_pivot['investasi'] = 0
    
    # Calculate net and saving rate
    monthly_pivot['net'] = monthly_pivot['pemasukan'] + monthly_pivot['investasi'] - monthly_pivot['pengeluaran']
    monthly_pivot['saving_rate'] = monthly_pivot['net'] / (monthly_pivot['pemasukan'] + monthly_pivot['investasi']) * 100
    monthly_pivot['saving_rate'] = monthly_pivot['saving_rate'].fillna(0)
    
    # Create sequences for LSTM
    sequence_length = 6
    
    # Features for prediction
    features = ['pemasukan', 'pengeluaran', 'investasi', 'is_early_year']
    scaler = StandardScaler()
    
    scaled_data = scaler.fit_transform(monthly_pivot[features])
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(monthly_pivot['pengeluaran'].iloc[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        # Not enough data for training
        return None, scaler, monthly_pivot, None
    
    # Train-test split
    split = int(len(X) * 0.8)
    if split == 0:
        split = 1
    
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build model
    model = Sequential([
        Bidirectional(GRU(64, return_sequences=True), input_shape=(sequence_length, len(features))),
        Dropout(0.2),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    # Train
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, y_test) if len(X_test) > 0 else None,
        callbacks=[early_stop] if len(X_test) > 0 else None,
        verbose=0
    )
    
    return model, scaler, monthly_pivot, history

def predict_future(model, scaler, monthly_pivot, months_ahead=1):
    """Predict future months"""
    if model is None:
        return [0] * months_ahead
    
    sequence_length = 6
    features = ['pemasukan', 'pengeluaran', 'investasi', 'is_early_year']
    
    # Get last sequence
    last_sequence = monthly_pivot[features].iloc[-sequence_length:].values
    last_sequence_scaled = scaler.transform(last_sequence)
    
    predictions = []
    current_sequence = last_sequence_scaled.copy()
    
    for _ in range(months_ahead):
        X_pred = current_sequence.reshape(1, sequence_length, len(features))
        pred = model.predict(X_pred, verbose=0)[0][0]
        predictions.append(pred)
        
        # Update sequence
        next_row = current_sequence[-1].copy()
        next_row[1] = pred
        current_sequence = np.vstack([current_sequence[1:], next_row])
    
    return predictions

# Main Dashboard
def main():
    st.markdown('<h1 class="main-header">💰 Dashboard Keuangan AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Klasifikasi Transaksi & Prediksi Pengeluaran dengan LSTM</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Memuat data...'):
        df = load_data()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/money--v1.png", width=80)
        st.markdown("## Navigasi")
        
        page = st.radio(
            "Pilih Halaman",
            ["📊 Overview Data", "🤖 Klasifikasi Transaksi", "📈 Prediksi Akhir Bulan", "💡 AI Insight Personal"]
        )
        
        st.markdown("---")
        st.markdown("### Filter Data")
        
        min_date = df['tanggal'].min().date()
        max_date = df['tanggal'].max().date()
        
        date_range = st.date_input(
            "Rentang Tanggal",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        category_filter = st.multiselect(
            "Kategori",
            options=df['kategori'].unique(),
            default=df['kategori'].unique()
        )
        
        st.markdown("---")
        st.markdown("### Tentang Dashboard")
        st.info(
            "Dashboard ini menggunakan:\n"
            "- **Random Forest** untuk klasifikasi transaksi\n"
            "- **LSTM/GRU** untuk prediksi tren bulanan\n"
            "- **Analisis personal** untuk perbandingan pengeluaran"
        )
    
    # Apply filters
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[
            (df['tanggal'].dt.date >= start_date) & 
            (df['tanggal'].dt.date <= end_date) &
            (df['kategori'].isin(category_filter))
        ]
    else:
        df_filtered = df[df['kategori'].isin(category_filter)]
    
    # Page content
    if page == "📊 Overview Data":
        show_overview(df_filtered, df)
    
    elif page == "🤖 Klasifikasi Transaksi":
        show_classification(df_filtered)
    
    elif page == "📈 Prediksi Akhir Bulan":
        show_prediction(df)
    
    else:
        show_insights(df_filtered, df)

def show_overview(df_filtered, df_full):
    st.markdown('<h2 class="sub-header">📊 Ringkasan Data Keuangan</h2>', unsafe_allow_html=True)
    
    # Calculate metrics
    total_pemasukan = df_filtered[df_filtered['type'] == 'Pemasukan']['jumlah'].sum() + df_filtered[df_filtered['kategori'] == 'Investasi']['jumlah'].sum()
    total_pengeluaran = df_filtered[df_filtered['type'] == 'Pengeluaran']['jumlah'].sum()
    net = total_pemasukan - total_pengeluaran
    saving_rate = (net / total_pemasukan * 100) if total_pemasukan > 0 else 0
    
    # Metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pemasukan", f"Rp {total_pemasukan:,.0f}")
    with col2:
        st.metric("Total Pengeluaran", f"Rp {total_pengeluaran:,.0f}")
    with col3:
        st.metric("Net Saldo", f"Rp {net:,.0f}", 
                  delta="Positif" if net > 0 else "Negatif")
    with col4:
        st.metric("Saving Rate", f"{saving_rate:.1f}%")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Tren Bulanan")
        # FIXED: Using pivot_table instead of agg with wrong column name
        monthly = df_filtered.pivot_table(
            index=df_filtered['tanggal'].dt.to_period('M'),
            columns='type',
            values='jumlah',
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        
        # Ensure columns exist
        if 'Pemasukan' not in monthly.columns:
            monthly['Pemasukan'] = 0
        if 'Pengeluaran' not in monthly.columns:
            monthly['Pengeluaran'] = 0
            
        monthly['bulan'] = monthly['tanggal'].astype(str)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=monthly['bulan'], y=monthly['Pemasukan'], name="Pemasukan", line=dict(color='green')),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=monthly['bulan'], y=monthly['Pengeluaran'], name="Pengeluaran", line=dict(color='red')),
            secondary_y=False
        )
        fig.update_layout(title="Tren Pemasukan vs Pengeluaran Bulanan", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🥧 Distribusi Kategori")
        category_sum = df_filtered[df_filtered['type'] == 'Pengeluaran'].groupby('kategori')['jumlah'].sum()
        if len(category_sum) > 0:
            fig = px.pie(values=category_sum.values, names=category_sum.index, title="Proporsi Pengeluaran per Kategori")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada data pengeluaran untuk ditampilkan")
    
    # Top transactions
    st.markdown("### 💰 Top 10 Transaksi Terbesar")
    top_transactions = df_filtered.nlargest(10, 'jumlah')[['tanggal', 'deskripsi', 'kategori', 'jumlah']]
    top_transactions['jumlah'] = top_transactions['jumlah'].apply(lambda x: f"Rp {x:,.0f}")
    st.dataframe(top_transactions, use_container_width=True)
    
    # Summary statistics
    st.markdown("### 📊 Statistik per Kategori")
    stats = df_filtered.groupby('kategori').agg({
        'jumlah': ['count', 'mean', 'sum']
    }).round(0)
    stats.columns = ['Jumlah Transaksi', 'Rata-rata', 'Total']
    stats['Rata-rata'] = stats['Rata-rata'].apply(lambda x: f"Rp {x:,.0f}")
    stats['Total'] = stats['Total'].apply(lambda x: f"Rp {x:,.0f}")
    st.dataframe(stats, use_container_width=True)

def show_classification(df):
    st.markdown('<h2 class="sub-header">🤖 Klasifikasi Transaksi Otomatis</h2>', unsafe_allow_html=True)
    
    st.info("""
    Model **Random Forest** digunakan untuk mengklasifikasikan transaksi berdasarkan deskripsi dan nominal.
    Model ini dilatih dengan ribuan transaksi untuk mengenali pola pengeluaran.
    """)
    
    # Train model
    with st.spinner('Melatih model klasifikasi...'):
        model, tfidf, le, accuracy = train_classification_model(df)
    
    # Show model performance
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Akurasi Model", f"{accuracy*100:.1f}%")
    with col2:
        st.metric("Jumlah Kategori", len(le.classes_))
    with col3:
        st.metric("Fitur yang Digunakan", "TF-IDF (100) + Nominal")
    
    st.markdown("---")
    
    # Test classification
    st.markdown("### 🧪 Coba Klasifikasi")
    
    col1, col2 = st.columns(2)
    with col1:
        test_desc = st.text_input("Deskripsi Transaksi", "beli makan siang")
    with col2:
        test_amount = st.number_input("Nominal (Rp)", min_value=0, value=35000, step=5000)
    
    if st.button("Klasifikasikan", type="primary"):
        # Transform input
        text_features = tfidf.transform([test_desc]).toarray()
        amount_features = np.array([[test_amount]])
        X_test = np.hstack([text_features, amount_features])
        
        # Predict
        pred = model.predict(X_test)[0]
        pred_category = le.inverse_transform([pred])[0]
        
        # Get probabilities
        probs = model.predict_proba(X_test)[0]
        prob_dict = {le.inverse_transform([i])[0]: probs[i] for i in range(len(le.classes_))}
        
        st.success(f"### Prediksi Kategori: **{pred_category}**")
        
        # Show confidence
        st.markdown("#### Tingkat Keyakinan per Kategori:")
        prob_df = pd.DataFrame([prob_dict]).T.reset_index()
        prob_df.columns = ['Kategori', 'Probabilitas']
        prob_df = prob_df.sort_values('Probabilitas', ascending=False)
        
        fig = px.bar(prob_df.head(5), x='Kategori', y='Probabilitas', 
                     title="Top 5 Prediksi Kategori", color='Probabilitas',
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Batch prediction example
    st.markdown("### 📋 Contoh Prediksi Batch")
    example_transactions = pd.DataFrame({
        'deskripsi': ['grab food malam', 'topup gopay', 'beli baju online', 'transfer ortu', 'bayar kos'],
        'jumlah': [45000, 50000, 150000, 1000000, 750000]
    })
    
    # Predict for examples
    text_features = tfidf.transform(example_transactions['deskripsi']).toarray()
    amount_features = example_transactions[['jumlah']].values
    X_example = np.hstack([text_features, amount_features])
    predictions = le.inverse_transform(model.predict(X_example))
    
    example_transactions['prediksi_kategori'] = predictions
    st.dataframe(example_transactions, use_container_width=True)

def show_prediction(df):
    st.markdown('<h2 class="sub-header">📈 Prediksi Pengeluaran Akhir Bulan</h2>', unsafe_allow_html=True)
    
    st.info("""
    Model **LSTM (Long Short-Term Memory)** dan **GRU (Gated Recurrent Unit)** digunakan untuk memprediksi pengeluaran bulanan.
    Model ini memperhatikan pola musiman termasuk **lonjakan awal tahun** (Januari-Februari) seperti THR, bonus, dan peningkatan belanja.
    """)
    
    # Train time series model
    with st.spinner('Melatih model LSTM/GRU... (mungkin memakan waktu 30-60 detik)'):
        model, scaler, monthly_data, history = train_timeseries_model(df)
    
    if model is None or len(monthly_data) < 7:
        st.warning("Data tidak mencukupi untuk training model time series. Minimal diperlukan 7 bulan data.")
        return
    
    # Show training performance
    if history is not None:
        st.markdown("### 📊 Performa Training Model")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss', line=dict(color='blue')))
            if 'val_loss' in history.history:
                fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss', line=dict(color='red')))
            fig.update_layout(title='Loss selama Training', xaxis_title='Epoch', yaxis_title='Loss')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history.history['mae'], name='Training MAE', line=dict(color='blue')))
            if 'val_mae' in history.history:
                fig.add_trace(go.Scatter(y=history.history['val_mae'], name='Validation MAE', line=dict(color='red')))
            fig.update_layout(title='MAE selama Training', xaxis_title='Epoch', yaxis_title='MAE')
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Make predictions
    st.markdown("### 🔮 Prediksi 3 Bulan ke Depan")
    
    # Predict next 3 months
    predictions = predict_future(model, scaler, monthly_data, months_ahead=3)
    
    # Get last actual values
    last_actual = monthly_data['pengeluaran'].iloc[-6:].values
    
    # Create forecast dataframe
    last_date = monthly_data['bulan'].iloc[-1]
    last_year = monthly_data['tahun'].iloc[-1]
    
    next_months = []
    next_years = []
    current_month = int(last_date)
    current_year = int(last_year)
    
    for i in range(3):
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
        next_months.append(current_month)
        next_years.append(current_year)
    
    forecast_df = pd.DataFrame({
        'bulan': next_months,
        'tahun': next_years,
        'prediksi_pengeluaran': predictions
    })
    forecast_df['bulan_nama'] = forecast_df.apply(
        lambda x: pd.Timestamp(year=int(x['tahun']), month=int(x['bulan']), day=1).strftime('%B %Y'), axis=1
    )
    
    # Show predictions
    col1, col2, col3 = st.columns(3)
    for i, col in enumerate([col1, col2, col3]):
        with col:
            st.metric(
                forecast_df['bulan_nama'].iloc[i],
                f"Rp {forecast_df['prediksi_pengeluaran'].iloc[i]:,.0f}",
                delta=None
            )
    
    # Visualization
    st.markdown("### 📈 Visualisasi Prediksi")
    
    # Combine historical and forecast
    historical_df = monthly_data[['bulan', 'tahun', 'pengeluaran']].copy()
    historical_df['bulan_nama'] = historical_df.apply(
        lambda x: pd.Timestamp(year=int(x['tahun']), month=int(x['bulan']), day=1).strftime('%b %Y'), axis=1
    )
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_df['bulan_nama'].iloc[-12:],
        y=historical_df['pengeluaran'].iloc[-12:],
        name='Historical Pengeluaran',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df['bulan_nama'],
        y=forecast_df['prediksi_pengeluaran'],
        name='Prediksi Pengeluaran',
        line=dict(color='red', width=2, dash='dash')
    ))
    fig.update_layout(
        title='Prediksi Pengeluaran Bulanan',
        xaxis_title='Bulan',
        yaxis_title='Pengeluaran (Rp)',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Early year pattern analysis
    st.markdown("### 🎯 Analisis Pola Awal Tahun")
    
    early_year_data = monthly_data[monthly_data['is_early_year'] == 1]
    normal_data = monthly_data[monthly_data['is_early_year'] == 0]
    
    col1, col2 = st.columns(2)
    with col1:
        if len(early_year_data) > 0:
            avg_early = early_year_data['pengeluaran'].mean()
            st.metric("Rata-rata Pengeluaran (Jan-Feb)", f"Rp {avg_early:,.0f}")
        else:
            st.metric("Rata-rata Pengeluaran (Jan-Feb)", "Data tidak cukup")
    
    with col2:
        if len(normal_data) > 0 and len(early_year_data) > 0:
            avg_normal = normal_data['pengeluaran'].mean()
            delta_pct = ((avg_early - avg_normal) / avg_normal * 100) if avg_normal > 0 else 0
            st.metric("Rata-rata Pengeluaran Bulan Lain", f"Rp {avg_normal:,.0f}",
                      delta=f"{delta_pct:+.1f}% vs awal tahun")
        else:
            st.metric("Rata-rata Pengeluaran Bulan Lain", "Data tidak cukup")
    
    st.caption("""
    **Insight**: Model LSTM/GRU memperhitungkan lonjakan pengeluaran pada awal tahun 
    (Januari-Februari) yang biasanya terjadi karena THR, bonus tahun baru, 
    dan peningkatan aktivitas belanja.
    """)

def show_insights(df_filtered, df_full):
    st.markdown('<h2 class="sub-header">💡 AI Insight Personal</h2>', unsafe_allow_html=True)
    
    st.info("""
    **Analisis Personal vs Rata-rata Pengguna**
    Bandingkan pola pengeluaran Anda dengan rata-rata pengguna untuk meningkatkan literasi keuangan.
    """)
    
    # Calculate user averages (only Pengeluaran)
    user_expenses = df_filtered[df_filtered['type'] == 'Pengeluaran']
    global_expenses = df_full[df_full['type'] == 'Pengeluaran']
    
    if len(user_expenses) > 0 and len(global_expenses) > 0:
        user_avg = user_expenses.groupby('kategori')['jumlah'].mean()
        global_avg = global_expenses.groupby('kategori')['jumlah'].mean()
        
        # Create comparison dataframe
        comparison = pd.DataFrame({
            'Kategori': user_avg.index,
            'Rata-rata Anda': user_avg.values,
            'Rata-rata Global': [global_avg.get(cat, 0) for cat in user_avg.index]
        })
        comparison['Selisih'] = comparison['Rata-rata Anda'] - comparison['Rata-rata Global']
        comparison['Persentase Selisih'] = (comparison['Selisih'] / comparison['Rata-rata Global'] * 100).fillna(0)
        comparison['Status'] = comparison['Persentase Selisih'].apply(
            lambda x: '🔴 Di Atas Rata-rata' if x > 10 
            else ('🟢 Di Bawah Rata-rata' if x < -10 else '🟡 Normal')
        )
        
        st.markdown("### 📊 Perbandingan Pengeluaran per Kategori")
        
        fig = px.bar(
            comparison, 
            x='Kategori', 
            y=['Rata-rata Anda', 'Rata-rata Global'],
            title='Perbandingan Rata-rata Transaksi per Kategori',
            barmode='group',
            color_discrete_map={'Rata-rata Anda': '#1E3A5F', 'Rata-rata Global': '#7BA7C9'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📝 Analisis Detail")
        
        for _, row in comparison.iterrows():
            if row['Status'] == '🔴 Di Atas Rata-rata':
                st.markdown(f"""
                <div class="insight-box">
                    <strong>{row['Kategori']}</strong><br>
                    💸 Anda menghabiskan <span style="color:red;">Rp {row['Selisih']:,.0f} lebih banyak</span> 
                    ({row['Persentase Selisih']:.1f}% di atas rata-rata) per transaksi.<br>
                    📌 <em>Saran: Evaluasi kembali kebutuhan dan cari alternatif yang lebih hemat.</em>
                </div>
                """, unsafe_allow_html=True)
            elif row['Status'] == '🟢 Di Bawah Rata-rata':
                st.markdown(f"""
                <div class="insight-box">
                    <strong>{row['Kategori']}</strong><br>
                    🎉 Anda menghemat <span style="color:green;">Rp {abs(row['Selisih']):,.0f}</span> 
                    ({abs(row['Persentase Selisih']):.1f}% di bawah rata-rata) per transaksi.<br>
                    📌 <em>Pertahankan kebiasaan baik ini!</em>
                </div>
                """, unsafe_allow_html=True)
        
        # Category with highest spending
        st.markdown("---")
        st.markdown("### 🎯 Rekomendasi Personal")
        
        top_category = user_avg.idxmax() if len(user_avg) > 0 else None
        top_amount = user_avg.max() if len(user_avg) > 0 else 0
        
        if top_category:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style="background-color:#f0f2f6; padding:1rem; border-radius:10px; text-align:center;">
                    <h3>⚠️ Perhatian</h3>
                    <p>Kategori dengan rata-rata transaksi tertinggi:</p>
                    <h2 style="color:#e74c3c;">{top_category}</h2>
                    <p>Rp {top_amount:,.0f} per transaksi</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Get month with highest spending
                category_data = df_filtered[df_filtered['kategori'] == top_category]
                if len(category_data) > 0:
                    peak_month = category_data.groupby(category_data['tanggal'].dt.month)['jumlah'].sum().idxmax()
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 
                                  'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']
                    
                    st.markdown(f"""
                    <div style="background-color:#e8f4f8; padding:1rem; border-radius:10px; text-align:center;">
                        <h3>📅 Pola Pengeluaran</h3>
                        <p>Puncak pengeluaran untuk <strong>{top_category}</strong> terjadi pada</p>
                        <h2 style="color:#2980b9;">{month_names[peak_month-1]}</h2>
                        <p>Perhatikan pola ini untuk perencanaan keuangan!</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Saving tips
        st.markdown("---")
        st.markdown("### 💡 Tips Menghemat")
        
        high_spending_cats = comparison[comparison['Persentase Selisih'] > 15]
        
        if len(high_spending_cats) > 0:
            st.markdown("Berdasarkan analisis Anda, perhatikan kategori berikut:")
            for _, row in high_spending_cats.iterrows():
                st.markdown(f"- **{row['Kategori']}**: Coba bandingkan harga di beberapa platform sebelum membeli")
        else:
            st.success("✅ Hebat! Pengeluaran Anda sudah efisien dibanding rata-rata pengguna. Pertahankan!")
        
        # Budget suggestion
        total_monthly = df_filtered[df_filtered['type'] == 'Pengeluaran']['jumlah'].sum()
        ideal_saving = total_monthly * 0.3
        
        if total_monthly > 0:
            st.markdown(f"""
            <div style="background-color:#2ecc71; padding:1rem; border-radius:10px; margin-top:1rem;">
                <h3 style="color:white; text-align:center;">🎯 Target Literasi Keuangan</h3>
                <p style="color:white; text-align:center;">
                    Berdasarkan data Anda, target menabung ideal bulan ini:<br>
                    <span style="font-size:1.5rem; font-weight:bold;">Rp {ideal_saving:,.0f}</span><br>
                    (30% dari total pengeluaran)
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Data tidak cukup untuk analisis perbandingan. Pastikan ada data pengeluaran.")

if __name__ == "__main__":
    main()