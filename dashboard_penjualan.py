import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from io import BytesIO

# Set page config
st.set_page_config(page_title="Dashboard Keuangan Personal", layout="wide")

# Load data
@st.cache_data
def load_data():
    # Baca file CSV
    df = pd.read_csv('dataset_transaksi.csv')
    
    # Generate tanggal dummy untuk demo (asumsi transaksi 2024-2025)
    np.random.seed(42)
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(len(df))]
    dates.sort()
    df['tanggal'] = dates
    
    # Generate nominal transaksi berdasarkan kategori
    kategori_harga = {
        'Investasi': np.random.uniform(500000, 5000000, len(df)),
        'Gaji': np.random.uniform(3000000, 10000000, len(df)),
        'Bonus': np.random.uniform(500000, 3000000, len(df)),
        'Belanja': np.random.uniform(50000, 500000, len(df)),
        'Transportasi': np.random.uniform(10000, 150000, len(df)),
        'Makanan & Minuman': np.random.uniform(15000, 100000, len(df)),
        'Hiburan': np.random.uniform(30000, 200000, len(df)),
        'Tagihan': np.random.uniform(100000, 1000000, len(df)),
        'Lainnya': np.random.uniform(10000, 200000, len(df))
    }
    
    nominal = []
    for _, row in df.iterrows():
        harga = kategori_harga.get(row['kategori'], np.random.uniform(10000, 1000000))
        nominal.append(harga[_])
    df['nominal'] = nominal
    
    # Tentukan tipe (pemasukan/pengeluaran)
    tipe = []
    for _, row in df.iterrows():
        if row['kategori'] in ['Gaji', 'Bonus', 'Investasi'] and 'profit' in row['deskripsi'].lower() or 'jual' in row['deskripsi'].lower() or 'pencairan' in row['deskripsi'].lower():
            tipe.append('Pemasukan')
        elif row['kategori'] in ['Gaji', 'Bonus']:
            tipe.append('Pemasukan')
        elif row['kategori'] in ['Investasi'] and ('profit' in row['deskripsi'].lower() or 'jual' in row['deskripsi'].lower() or 'pencairan' in row['deskripsi'].lower()):
            tipe.append('Pemasukan')
        else:
            tipe.append('Pengeluaran')
    df['tipe'] = tipe
    
    return df

# Prepare time series data
def prepare_timeseries(df, lookback=30):
    df_daily = df.groupby(pd.Grouper(key='tanggal', freq='D')).agg({
        'nominal': lambda x: x[df['tipe'] == 'Pengeluaran'].sum() - x[df['tipe'] == 'Pemasukan'].sum()
    }).fillna(0)
    df_daily.columns = ['saldo_bersih']
    df_daily['pengeluaran'] = df[df['tipe'] == 'Pengeluaran'].groupby(pd.Grouper(key='tanggal', freq='D'))['nominal'].sum().fillna(0)
    df_daily['pemasukan'] = df[df['tipe'] == 'Pemasukan'].groupby(pd.Grouper(key='tanggal', freq='D'))['nominal'].sum().fillna(0)
    
    return df_daily

# Build LSTM/GRU model
def build_model(input_shape, model_type='LSTM'):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=False))
    else:  # GRU
        model.add(GRU(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Train and predict
def train_and_predict(df_daily, lookback=30, model_type='LSTM'):
    # Tambah fitur lonjakan awal tahun (Januari-Maret)
    df_daily['month'] = df_daily.index.month
    df_daily['early_year_spike'] = df_daily['month'].isin([1, 2, 3]).astype(float)
    df_daily['pengeluaran_ma7'] = df_daily['pengeluaran'].rolling(7).mean().fillna(df_daily['pengeluaran'])
    df_daily['pemasukan_ma7'] = df_daily['pemasukan'].rolling(7).mean().fillna(df_daily['pemasukan'])
    
    features = ['pengeluaran', 'pemasukan', 'early_year_spike', 'pengeluaran_ma7', 'pemasukan_ma7']
    target = 'saldo_bersih'
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(df_daily[features])
    y_scaled = scaler_y.fit_transform(df_daily[[target]])
    
    X, y = [], []
    for i in range(lookback, len(X_scaled)):
        X.append(X_scaled[i-lookback:i])
        y.append(y_scaled[i])
    
    X, y = np.array(X), np.array(y)
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build and train model
    model = build_model((lookback, len(features)), model_type)
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, 
                        validation_data=(X_test, y_test), verbose=0)
    
    # Predict next 30 days
    last_sequence = X_scaled[-lookback:]
    predictions = []
    for _ in range(30):
        pred = model.predict(last_sequence.reshape(1, lookback, len(features)), verbose=0)
        predictions.append(pred[0, 0])
        # Update sequence
        new_row = last_sequence[-1].copy()
        last_sequence = np.vstack([last_sequence[1:], new_row])
        last_sequence[-1, 0] = pred[0, 0]  # Update pengeluaran
    
    predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Predict on test
    y_pred = model.predict(X_test, verbose=0)
    y_pred_actual = scaler_y.inverse_transform(y_pred)
    y_test_actual = scaler_y.inverse_transform(y_test)
    
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    
    return model, predictions, mae, rmse, history, df_daily, scaler_X, scaler_y

# Generate category insights
def get_category_insights(df, selected_category=None):
    if selected_category:
        df_cat = df[df['kategori'] == selected_category]
        df_other = df[df['kategori'] != selected_category]
        user_avg = df_cat[df_cat['tipe'] == 'Pengeluaran']['nominal'].mean()
        overall_avg = df_other[df_other['tipe'] == 'Pengeluaran']['nominal'].mean()
        percentage = ((user_avg - overall_avg) / overall_avg) * 100 if overall_avg > 0 else 0
        return user_avg, overall_avg, percentage
    else:
        # Top spending categories
        spending_by_cat = df[df['tipe'] == 'Pengeluaran'].groupby('kategori')['nominal'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
        income_by_cat = df[df['tipe'] == 'Pemasukan'].groupby('kategori')['nominal'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
        return spending_by_cat, income_by_cat

# Main dashboard
def main():
    st.title("💰 Dashboard Prediksi Keuangan Personal")
    st.markdown("### Analisis & Prediksi Cash Flow dengan LSTM/GRU")
    
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.header("⚙️ Konfigurasi")
    model_type = st.sidebar.selectbox("Pilih Model", ["LSTM", "GRU"])
    lookback = st.sidebar.slider("Lookback Days", 7, 60, 30)
    
    # Prepare data
    df_daily = prepare_timeseries(df)
    
    # Train model
    with st.spinner(f"Training {model_type} model..."):
        model, predictions, mae, rmse, history, df_daily, scaler_X, scaler_y = train_and_predict(df_daily, lookback, model_type)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    total_pengeluaran = df[df['tipe'] == 'Pengeluaran']['nominal'].sum()
    total_pemasukan = df[df['tipe'] == 'Pemasukan']['nominal'].sum()
    saldo_akhir = total_pemasukan - total_pengeluaran
    pred_next_month = predictions.sum()
    
    col1.metric("Total Pemasukan", f"Rp {total_pemasukan:,.0f}")
    col2.metric("Total Pengeluaran", f"Rp {total_pengeluaran:,.0f}")
    col3.metric("Saldo Bersih", f"Rp {saldo_akhir:,.0f}", 
                delta=f"{saldo_akhir/total_pemasukan*100:.1f}%" if total_pemasukan > 0 else "")
    col4.metric("Prediksi Bulan Depan", f"Rp {pred_next_month:,.0f}",
                delta="Negatif" if pred_next_month < 0 else "Positif")
    
    # Chart: Historical vs Prediction
    st.subheader("📈 Prediksi Cash Flow 30 Hari ke Depan")
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot historical
    last_90 = df_daily['saldo_bersih'].iloc[-90:]
    ax.plot(last_90.index, last_90.values, label='Historical (90 hari)', color='blue', linewidth=2)
    
    # Plot predictions
    last_date = df_daily.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(30)]
    ax.plot(future_dates, predictions, label='Prediksi (30 hari)', color='red', linestyle='--', linewidth=2)
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Saldo Bersih (Rp)')
    ax.set_title(f'Prediksi dengan Model {model_type} (MAE: Rp {mae:,.0f}, RMSE: Rp {rmse:,.0f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Training loss
    st.subheader("📉 Training History")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (MSE)')
    ax2.set_title('Model Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    
    # Category Insights
    st.subheader("📊 Insight Personal: Perbandingan Pengeluaran")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        categories = sorted(df['kategori'].unique())
        selected_cat = st.selectbox("Pilih Kategori untuk Analisis", categories)
    
    user_avg, overall_avg, percentage = get_category_insights(df, selected_cat)
    
    with col2:
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        bars = ax3.bar(['Rata-rata Anda', 'Rata-rata Global'], [user_avg, overall_avg], 
                       color=['#FF6B6B', '#4ECDC4'])
        ax3.set_ylabel('Rata-rata Pengeluaran (Rp)')
        ax3.set_title(f'Perbandingan Pengeluaran - Kategori: {selected_cat}')
        ax3.bar_label(bars, fmt='Rp %.0f')
        
        if percentage > 0:
            ax3.annotate(f'Anda {percentage:.1f}% lebih tinggi dari rata-rata', 
                        xy=(0.5, 0.95), xycoords='axes fraction', ha='center',
                        fontsize=10, color='red')
            st.warning(f"⚠️ Anda menghabiskan **{percentage:.1f}% lebih banyak** dari rata-rata pengguna untuk kategori **{selected_cat}**")
        elif percentage < 0:
            ax3.annotate(f'Anda {abs(percentage):.1f}% lebih hemat dari rata-rata', 
                        xy=(0.5, 0.95), xycoords='axes fraction', ha='center',
                        fontsize=10, color='green')
            st.success(f"✅ Anda **{abs(percentage):.1f}% lebih hemat** dari rata-rata pengguna untuk kategori **{selected_cat}**")
        else:
            st.info(f"📊 Pengeluaran Anda untuk kategori **{selected_cat}** sebanding dengan rata-rata pengguna")
        
        st.pyplot(fig3)
    
    # Top Categories
    st.subheader("🏷️ Top Kategori Pengeluaran")
    spending_by_cat, income_by_cat = get_category_insights(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔴 Pengeluaran Terbesar**")
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        top_spending = spending_by_cat.head(8)
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_spending)))
        ax4.barh(top_spending.index, top_spending['sum'], color=colors)
        ax4.set_xlabel('Total Pengeluaran (Rp)')
        ax4.set_title('Top 8 Kategori Pengeluaran')
        st.pyplot(fig4)
    
    with col2:
        st.markdown("**🟢 Pemasukan Terbesar**")
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        top_income = income_by_cat.head(8)
        colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(top_income)))
        ax5.barh(top_income.index, top_income['sum'], color=colors)
        ax5.set_xlabel('Total Pemasukan (Rp)')
        ax5.set_title('Top 8 Kategori Pemasukan')
        st.pyplot(fig5)
    
    # Monthly trend
    st.subheader("📅 Tren Bulanan")
    df['bulan'] = df['tanggal'].dt.to_period('M')
    monthly = df.groupby(['bulan', 'tipe'])['nominal'].sum().unstack().fillna(0)
    
    fig6, ax6 = plt.subplots(figsize=(12, 5))
    monthly['Pengeluaran'].plot(kind='bar', ax=ax6, color='red', alpha=0.7, label='Pengeluaran')
    monthly['Pemasukan'].plot(kind='bar', ax=ax6, color='green', alpha=0.7, label='Pemasukan')
    ax6.set_xlabel('Bulan')
    ax6.set_ylabel('Nominal (Rp)')
    ax6.set_title('Tren Bulanan Pemasukan vs Pengeluaran')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig6)
    
    # Early year spike insight
    st.subheader("🎯 Insight: Lonjakan Awal Tahun")
    early_months = df[df['tanggal'].dt.month.isin([1, 2, 3])]
    early_spending = early_months[early_months['tipe'] == 'Pengeluaran']['nominal'].mean()
    other_spending = df[~df['tanggal'].dt.month.isin([1, 2, 3])][df['tipe'] == 'Pengeluaran']['nominal'].mean()
    
    col1, col2 = st.columns(2)
    col1.metric("Rata-rata Pengeluaran (Jan-Mar)", f"Rp {early_spending:,.0f}")
    col2.metric("Rata-rata Pengeluaran (Apr-Des)", f"Rp {other_spending:,.0f}",
                delta=f"{(early_spending-other_spending)/other_spending*100:.1f}%")
    
    if early_spending > other_spending:
        st.info("💡 **Insight:** Anda cenderung memiliki pengeluaran lebih tinggi di awal tahun. Model LSTM/GRU telah mempertimbangkan pola ini untuk prediksi yang lebih akurat.")
    
    # Data preview
    with st.expander("📋 Preview Data Transaksi"):
        st.dataframe(df.head(100))
    
    # Download predictions
    pred_df = pd.DataFrame({
        'Tanggal': future_dates,
        'Prediksi_Saldo': predictions.flatten()
    })
    csv = pred_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Prediksi (CSV)", csv, "prediksi_keuangan.csv", "text/csv")

if __name__ == "__main__":
    main()