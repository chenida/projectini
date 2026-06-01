import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Prophet & XGBoost
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Dashboard Keuangan GenZ - Model B",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6; border-radius: 8px; padding: 8px 16px;
    }
    .highlight-box {
        background: #e8f4fd; border-left: 4px solid #1f77b4;
        padding: 1rem; border-radius: 4px; margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# 1. LOAD & PREPROCESS DATA
# ======================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset.csv')
    except FileNotFoundError:
        st.error("File dataset.csv tidak ditemukan!")
        return pd.DataFrame()
    
    # Clean numeric columns
    for col in ['Harga_Satuan', 'Total_Penjualan']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('Rp', '').str.replace('.', '').str.replace(',', '').astype(float)
    
    # Rename Total_Penjualan to Total
    if 'Total_Penjualan' in df.columns:
        df = df.rename(columns={'Total_Penjualan': 'Total'})
    
    # Parse Tanggal
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
    df = df.dropna(subset=['Tanggal'])
    df = df[(df['Tanggal'] >= '2023-01-01') & (df['Tanggal'] <= '2026-12-31')]
    
    # Add column type
    df['Tipe_Transaksi'] = 'Pengeluaran'
    
    return df

@st.cache_data
def generate_income_data():
    """Generate realistic income data for GenZ"""
    dates = pd.date_range(start='2023-01-01', end='2026-05-31', freq='D')
    
    income_records = []
    
    # Regular salary (monthly, around 25th)
    for date in dates:
        if date.day == 25:
            salary = np.random.randint(4000000, 8000000)
            income_records.append({
                'Tanggal': date,
                'Kategori': 'Gaji',
                'Produk': 'Gaji Bulanan',
                'Jumlah': 1,
                'Harga_Satuan': salary,
                'Total': salary,
                'Tipe_Transaksi': 'Pemasukan',
                'Sumber': 'Gaji'
            })
    
    # Weekend freelance (Saturday)
    for date in dates:
        if date.weekday() == 5:
            freelance = np.random.randint(200000, 800000)
            income_records.append({
                'Tanggal': date,
                'Kategori': 'Freelance',
                'Produk': 'Freelance',
                'Jumlah': 1,
                'Harga_Satuan': freelance,
                'Total': freelance,
                'Tipe_Transaksi': 'Pemasukan',
                'Sumber': 'Freelance'
            })
    
    # Kiriman orang tua (1st and 15th)
    for date in dates:
        if date.day == 1 or date.day == 15:
            kiriman = np.random.randint(500000, 1500000)
            income_records.append({
                'Tanggal': date,
                'Kategori': 'Kiriman Keluarga',
                'Produk': 'Kiriman Orang Tua',
                'Jumlah': 1,
                'Harga_Satuan': kiriman,
                'Total': kiriman,
                'Tipe_Transaksi': 'Pemasukan',
                'Sumber': 'Keluarga'
            })
    
    # Bonus/THR (June and December)
    for date in dates:
        if (date.month == 6 and date.day == 20) or (date.month == 12 and date.day == 20):
            bonus = np.random.randint(1000000, 3000000)
            income_records.append({
                'Tanggal': date,
                'Kategori': 'Bonus',
                'Produk': 'Bonus/THR',
                'Jumlah': 1,
                'Harga_Satuan': bonus,
                'Total': bonus,
                'Tipe_Transaksi': 'Pemasukan',
                'Sumber': 'Bonus'
            })
    
    return pd.DataFrame(income_records)

@st.cache_data
def load_and_combine_data():
    df_expense = load_data()
    if df_expense.empty:
        return pd.DataFrame()
    
    df_income = generate_income_data()
    
    # Combine
    df_combined = pd.concat([df_expense, df_income], ignore_index=True)
    df_combined['Tanggal'] = pd.to_datetime(df_combined['Tanggal'])
    
    return df_combined

df = load_and_combine_data()

if df.empty:
    st.error("Tidak dapat memuat data. Pastikan file dataset.csv ada.")
    st.stop()

# ======================
# 2. AGREGASI TIME SERIES
# ======================
@st.cache_data
def get_daily_series(_df):
    # Pengeluaran harian
    expense_data = _df[_df['Tipe_Transaksi'] == 'Pengeluaran']
    if not expense_data.empty:
        expense_daily = expense_data.groupby('Tanggal')['Total'].sum().reset_index()
        expense_daily = expense_daily.set_index('Tanggal').resample('D').sum().fillna(0)
        expense_daily.columns = ['Pengeluaran']
    else:
        expense_daily = pd.DataFrame(columns=['Pengeluaran'])
    
    # Pemasukan harian
    income_data = _df[_df['Tipe_Transaksi'] == 'Pemasukan']
    if not income_data.empty:
        income_daily = income_data.groupby('Tanggal')['Total'].sum().reset_index()
        income_daily = income_daily.set_index('Tanggal').resample('D').sum().fillna(0)
        income_daily.columns = ['Pemasukan']
    else:
        income_daily = pd.DataFrame(columns=['Pemasukan'])
    
    # Combine
    daily = expense_daily.join(income_daily, how='outer').fillna(0)
    daily['Net_Cashflow'] = daily['Pemasukan'] - daily['Pengeluaran']
    
    return daily

@st.cache_data
def get_category_expense(_df):
    """Get expense breakdown by category"""
    expense_df = _df[_df['Tipe_Transaksi'] == 'Pengeluaran'].copy()
    
    if expense_df.empty:
        return pd.DataFrame(), pd.DataFrame(), expense_df
    
    # Group by category
    category_total = expense_df.groupby('Kategori')['Total'].agg(['sum', 'count']).reset_index()
    category_total.columns = ['Kategori', 'Total_Pengeluaran', 'Jumlah_Transaksi']
    category_total = category_total.sort_values('Total_Pengeluaran', ascending=False)
    
    # Monthly category breakdown
    expense_df['Bulan'] = expense_df['Tanggal'].dt.to_period('M')
    monthly_category = expense_df.groupby(['Bulan', 'Kategori'])['Total'].sum().reset_index()
    monthly_category['Bulan'] = monthly_category['Bulan'].dt.to_timestamp()
    
    return category_total, monthly_category, expense_df

daily_cashflow = get_daily_series(df)
category_total, monthly_category, expense_df = get_category_expense(df)

# ======================
# 3. FEATURE ENGINEERING
# ======================
def create_advanced_features(series: pd.Series) -> pd.DataFrame:
    """Feature engineering untuk XGBoost."""
    df_feat = series.to_frame(name='y')
    df_feat['ds'] = df_feat.index

    df_feat['dayofweek'] = df_feat['ds'].dt.dayofweek
    df_feat['dayofmonth'] = df_feat['ds'].dt.day
    df_feat['month'] = df_feat['ds'].dt.month
    df_feat['quarter'] = df_feat['ds'].dt.quarter
    df_feat['year'] = df_feat['ds'].dt.year
    df_feat['is_weekend'] = (df_feat['dayofweek'] >= 5).astype(int)
    df_feat['is_month_end'] = df_feat['ds'].dt.is_month_end.astype(int)
    df_feat['is_month_start'] = df_feat['ds'].dt.is_month_start.astype(int)

    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    df_feat['dow_sin'] = np.sin(2 * np.pi * df_feat['dayofweek'] / 7)
    df_feat['dow_cos'] = np.cos(2 * np.pi * df_feat['dayofweek'] / 7)

    for lag in [1, 7, 14, 30]:
        df_feat[f'lag_{lag}'] = df_feat['y'].shift(lag)

    for window in [7, 14, 30]:
        df_feat[f'rolling_mean_{window}'] = df_feat['y'].rolling(window).mean()
        df_feat[f'rolling_std_{window}'] = df_feat['y'].rolling(window).std()

    df_feat['trend'] = np.arange(len(df_feat))

    return df_feat.dropna()

# ======================
# 4. TRAIN PROPHET MODEL (DIPERBAIKI)
# ======================
@st.cache_resource
def train_prophet(daily_cashflow):
    # 1. Pastikan kita membuat copy agar tidak merusak data asli
    prophet_df = daily_cashflow.copy()
    
    # 2. Kasus A: Jika tanggal kamu saat ini berstatus sebagai INDEX di DataFrame
    if prophet_df.index.name == 'Tanggal' or isinstance(prophet_df.index, pd.DatetimeIndex):
        prophet_df = prophet_df.reset_index()  # Menjadikan index sebagai kolom biasa
    
    # 3. Ganti nama kolom sesuai kemauan Prophet ('ds' dan 'y')
    # Sesuaikan 'Tanggal Asli' dan 'Nilai Asli' dengan nama kolom di daily_cashflow kamu
    prophet_df = prophet_df.rename(columns={
        'Tanggal': 'ds',        # Ganti 'Tanggal' dengan nama kolom tanggalmu
        'Net_Cashflow': 'y'     # Ganti 'Net_Cashflow' dengan kolom yang ingin diprediksi
    })
    
    # 4. Pastikan kolom 'ds' benar-benar bertipe datetime
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    # 5. Inisialisasi dan Training Model
    m = Prophet()
    m.fit(prophet_df)  # Sekarang ini tidak akan error lagi
    
    return m

# ======================
# 5. TRAIN XGBOOST MODEL
# ======================
@st.cache_resource
def train_xgboost(_daily):
    feat_df = create_advanced_features(_daily['Net_Cashflow'])
    feature_cols = [c for c in feat_df.columns if c not in ['y', 'ds']]
    X = feat_df[feature_cols].values
    y = feat_df['y'].values

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100

    return model, feat_df, feature_cols, mae, rmse, mape, y_test, y_pred

# Train models only if data is available
if not daily_cashflow.empty and len(daily_cashflow) > 30:
    with st.spinner("🔄 Training model..."):
        prophet_model = train_prophet(daily_cashflow)
        xgb_model, feat_df, feature_cols, xgb_mae, xgb_rmse, xgb_mape, y_test_xgb, y_pred_xgb = train_xgboost(daily_cashflow)
    
    split_idx = int(0.8 * len(daily_cashflow))
    test_dates = daily_cashflow.index[split_idx:]
    
    # Prepare data for Prophet evaluation
    prophet_eval_df = daily_cashflow.copy()
    if prophet_eval_df.index.name == 'Tanggal' or isinstance(prophet_eval_df.index, pd.DatetimeIndex):
        prophet_eval_df = prophet_eval_df.reset_index()
    prophet_eval_df = prophet_eval_df.rename(columns={'Tanggal': 'ds', 'Net_Cashflow': 'y'})
    prophet_eval_df['ds'] = pd.to_datetime(prophet_eval_df['ds'])
    
    prophet_forecast = prophet_model.predict(prophet_eval_df)
    prophet_test_pred = prophet_forecast.set_index('ds').loc[test_dates, 'yhat'].values
    prophet_test_true = daily_cashflow['Net_Cashflow'].iloc[split_idx:].values
    p_mae = mean_absolute_error(prophet_test_true, prophet_test_pred)
    p_rmse = np.sqrt(mean_squared_error(prophet_test_true, prophet_test_pred))
    p_mape = np.mean(np.abs((prophet_test_true - prophet_test_pred) / (prophet_test_true + 1e-6))) * 100
    
    def prophet_future_predict(model, days=30):
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
        return forecast.tail(days)
    
    def xgb_future_predict(model, daily, feature_cols, days=30):
        extended = daily['Net_Cashflow'].copy()
        predictions = []
        future_dates = pd.date_range(start=daily.index[-1] + timedelta(days=1), periods=days)
        for fd in future_dates:
            feat = create_advanced_features(extended)
            if len(feat) == 0:
                predictions.append(extended.mean())
                extended.loc[fd] = extended.mean()
                continue
            last_feat = feat[feature_cols].iloc[[-1]].values
            pred = float(model.predict(last_feat)[0])
            predictions.append(pred)
            extended.loc[fd] = pred
        return future_dates, np.array(predictions)
    
    prophet_future = prophet_future_predict(prophet_model, days=30)
    xgb_future_dates, xgb_future_pred = xgb_future_predict(xgb_model, daily_cashflow, feature_cols, days=30)
    
    model_trained = True
else:
    model_trained = False
    st.warning("⚠️ Data tidak cukup untuk training model. Minimal diperlukan 30 hari data.")

# ======================
# SIDEBAR
# ======================
st.sidebar.image("https://img.icons8.com/fluency/96/000000/money-bag.png", width=60)
st.sidebar.title("💰 Dashboard Keuangan GenZ")
st.sidebar.markdown("**Model B: Prophet + XGBoost**")
st.sidebar.markdown("---")

# Date filter
date_min = daily_cashflow.index.min().date() if not daily_cashflow.empty else datetime(2023,1,1).date()
date_max = daily_cashflow.index.max().date() if not daily_cashflow.empty else datetime(2026,5,31).date()

start_date = st.sidebar.date_input("Mulai Tanggal", date_min)
end_date = st.sidebar.date_input("Akhir Tanggal", date_max)

# Category filter for expense analysis
st.sidebar.markdown("---")
st.sidebar.markdown("**📂 Filter Kategori Pengeluaran**")
all_categories = sorted(category_total['Kategori'].unique()) if not category_total.empty else []
selected_categories = st.sidebar.multiselect(
    "Pilih Kategori:",
    options=all_categories,
    default=all_categories[:5] if len(all_categories) > 5 else all_categories
)

forecast_days = st.sidebar.slider("Horizon Prediksi (hari)", 7, 90, 30)
selected_model = st.sidebar.radio("Tampilkan Prediksi Model:", ["Prophet", "XGBoost", "Ensemble"])

# Filter data by date
filtered_cashflow = daily_cashflow[
    (daily_cashflow.index >= pd.Timestamp(start_date)) &
    (daily_cashflow.index <= pd.Timestamp(end_date))
] if not daily_cashflow.empty else pd.DataFrame()

# Filter expense data by selected categories
if not expense_df.empty:
    filtered_expense = expense_df[expense_df['Kategori'].isin(selected_categories)] if selected_categories else expense_df
    filtered_expense = filtered_expense[
        (filtered_expense['Tanggal'] >= pd.Timestamp(start_date)) &
        (filtered_expense['Tanggal'] <= pd.Timestamp(end_date))
    ]
else:
    filtered_expense = pd.DataFrame()

# ======================
# HEADER
# ======================
st.title("💰 Dashboard Keuangan GenZ — Model B: Prophet + XGBoost")
st.markdown("**Analisis Pengeluaran per Kategori + Prediksi Net Cashflow**")
st.markdown("---")

# KPI Row
col1, col2, col3, col4, col5 = st.columns(5)

total_expense = df[df['Tipe_Transaksi'] == 'Pengeluaran']['Total'].sum() if not df.empty else 0
total_income = df[df['Tipe_Transaksi'] == 'Pemasukan']['Total'].sum() if not df.empty else 0
net_cashflow = total_income - total_expense

if not daily_cashflow.empty and (daily_cashflow.index.max() - daily_cashflow.index.min()).days > 0:
    avg_monthly_expense = total_expense / ((daily_cashflow.index.max() - daily_cashflow.index.min()).days / 30.44)
else:
    avg_monthly_expense = 0

top_category = category_total.iloc[0]['Kategori'] if not category_total.empty else "N/A"

# Prediksi future
if model_trained:
    if selected_model == "Prophet":
        prophet_30 = prophet_future_predict(prophet_model, days=forecast_days)
        avg_pred_net = float(prophet_30['yhat'].mean())
    elif selected_model == "XGBoost":
        _, xgb_pred_30 = xgb_future_predict(xgb_model, daily_cashflow, feature_cols, days=forecast_days)
        avg_pred_net = float(xgb_pred_30.mean())
    else:
        p30 = float(prophet_future_predict(prophet_model, days=forecast_days)['yhat'].mean())
        _, x30 = xgb_future_predict(xgb_model, daily_cashflow, feature_cols, days=forecast_days)
        avg_pred_net = (p30 + float(x30.mean())) / 2
else:
    avg_pred_net = 0

with col1:
    st.metric("💰 Total Pemasukan", f"Rp{total_income/1e6:.1f}Jt")
with col2:
    st.metric("💸 Total Pengeluaran", f"Rp{total_expense/1e6:.1f}Jt")
with col3:
    color = "normal" if net_cashflow >= 0 else "inverse"
    st.metric("📊 Net Cashflow", f"Rp{net_cashflow/1e6:+.1f}Jt", delta_color=color)
with col4:
    st.metric("🏆 Top Kategori", top_category)
with col5:
    st.metric(f"🔮 Prediksi Net ({forecast_days}H)", f"Rp{avg_pred_net/1e3:,.0f}K")

st.markdown("---")

# ======================
# TABS
# ======================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Prediksi Cashflow",
    "📊 Analisis per Kategori",
    "🧪 A/B Testing",
    "📅 Tren & Pola",
    "📋 Ringkasan"
])

# ---- TAB 1: PREDIKSI ----
with tab1:
    st.subheader(f"Prediksi Net Cashflow {forecast_days} Hari ke Depan — {selected_model}")
    
    if model_trained:
        if selected_model in ["Prophet", "Ensemble"]:
            pf = prophet_future_predict(prophet_model, days=forecast_days)
        if selected_model in ["XGBoost", "Ensemble"]:
            xfd, xfp = xgb_future_predict(xgb_model, daily_cashflow, feature_cols, days=forecast_days)
        
        fig = go.Figure()
        
        # Historical
        mask = (daily_cashflow.index >= pd.Timestamp(start_date)) & (daily_cashflow.index <= pd.Timestamp(end_date))
        hist = daily_cashflow[mask]
        
        if not hist.empty:
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist['Pengeluaran'],
                name='Pengeluaran', fill='tozeroy',
                line=dict(color='#f5576c', width=1),
                fillcolor='rgba(245,87,108,0.3)'
            ))
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist['Pemasukan'],
                name='Pemasukan', fill='tonexty',
                line=dict(color='#4facfe', width=1),
                fillcolor='rgba(79,172,254,0.3)'
            ))
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist['Net_Cashflow'],
                mode='lines', name='Net Cashflow',
                line=dict(color='#43e97b', width=2.5)
            ))
        
        # Forecast
        if selected_model in ["Prophet", "Ensemble"]:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(pf['ds']), y=pf['yhat'],
                mode='lines+markers', name='Prophet Forecast',
                line=dict(color='#FF5722', dash='dash')
            ))
        if selected_model in ["XGBoost", "Ensemble"]:
            fig.add_trace(go.Scatter(
                x=xfd, y=xfp,
                mode='lines+markers', name='XGBoost Forecast',
                line=dict(color='#4CAF50', dash='dot')
            ))
        if selected_model == "Ensemble":
            ensemble = (pf['yhat'].values + xfp) / 2
            fig.add_trace(go.Scatter(
                x=xfd, y=ensemble,
                mode='lines', name='Ensemble Forecast',
                line=dict(color='#9C27B0', width=2.5)
            ))
        
# ATAU JIKA INGIN TETAP BERBENTUK OBJEK TANGGAL, GUNAKAN INI:
# GANTI BARIS 518 MENJADI SEPERTI INI:
 #  NEW (Pass the actual date/timestamp object)
# 1. Gambar garis vertikal saja (Tanpa parameter annotation bawaan)
fig.add_vline(
    x=daily_cashflow.index[-1], 
    line_dash="dot", 
    line_color="gray"
)

# 2. Tambahkan teks "Hari Ini" secara manual di atas garis tersebut
fig.add_annotation(
    x=daily_cashflow.index[-1], # Posisi X mengikuti tanggal terakhir
    y=1,                        # Angka 1 berarti di bagian paling atas grafik
    yref="paper",               # Mengunci posisi Y di canvas grafik (bukan berdasarkan nilai nominal Rp)
    text="Hari Ini",
    showarrow=False,            # Menghilangkan panah penunjuk
    xanchor="right",            # Posisi teks rata kanan dari garis vertikal
    yanchor="top",              # Posisi teks nempel di batas atas grafik
    font=dict(color="gray")     # Menyamakan warna teks dengan warna garis
)

# 3. Update layout dan tampilkan ke Streamlit
fig.update_layout(
    title="Cashflow Harian + Prediksi", 
    xaxis_title="Tanggal",
    yaxis_title="Nominal (Rp)", 
    hovermode='x unified', 
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# ---- TAB 2: ANALISIS PER KATEGORI ----
with tab2:
    st.subheader("📊 Analisis Pengeluaran per Kategori")
    
    if not category_total.empty:
        # Top categories overview
        st.markdown("### Top Kategori Pengeluaran")
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            top10 = category_total.head(10)
            fig_top = px.bar(top10, x='Total_Pengeluaran', y='Kategori', orientation='h',
                             title="Top 10 Kategori Pengeluaran (Total)",
                             labels={'Total_Pengeluaran': 'Total Pengeluaran (Rp)', 'Kategori': ''},
                             color='Total_Pengeluaran', color_continuous_scale='Reds')
            fig_top.update_layout(height=400)
            st.plotly_chart(fig_top, use_container_width=True)
        
        with col_c2:
            fig_pie = px.pie(category_total.head(8), values='Total_Pengeluaran', names='Kategori',
                             title="Proporsi Pengeluaran per Kategori", hole=0.4)
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Category details table
        st.markdown("### Detail Pengeluaran per Kategori")
        st.dataframe(
            category_total.style.format({
                'Total_Pengeluaran': 'Rp{:,.0f}',
                'Jumlah_Transaksi': '{:,.0f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Monthly category trend
        if not monthly_category.empty and selected_categories:
            st.markdown("### Tren Bulanan per Kategori")
            filtered_monthly = monthly_category[monthly_category['Kategori'].isin(selected_categories)]
            
            if not filtered_monthly.empty:
                fig_monthly_cat = px.area(filtered_monthly, x='Bulan', y='Total', color='Kategori',
                                          title="Pengeluaran Bulanan per Kategori",
                                          labels={'Total': 'Pengeluaran (Rp)', 'Bulan': 'Bulan'},
                                          color_discrete_sequence=px.colors.qualitative.Set2)
                fig_monthly_cat.update_layout(height=450)
                st.plotly_chart(fig_monthly_cat, use_container_width=True)
        
        # Detailed breakdown for selected category
        st.markdown("### Breakdown Detail Kategori")
        selected_detail_cat = st.selectbox("Pilih Kategori untuk Detail:", all_categories)
        
        if selected_detail_cat and not expense_df.empty:
            cat_detail = expense_df[expense_df['Kategori'] == selected_detail_cat]
            
            if not cat_detail.empty:
                # Top products in this category
                product_summary = cat_detail.groupby('Produk')['Total'].agg(['sum', 'count']).reset_index()
                product_summary = product_summary.sort_values('sum', ascending=False).head(10)
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    fig_products = px.bar(product_summary, x='sum', y='Produk', orientation='h',
                                          title=f"Top 10 Produk - {selected_detail_cat}",
                                          labels={'sum': 'Total Pengeluaran (Rp)'},
                                          color='sum', color_continuous_scale='Teal')
                    st.plotly_chart(fig_products, use_container_width=True)
                
                with col_d2:
                    cat_monthly = cat_detail.groupby(cat_detail['Tanggal'].dt.to_period('M'))['Total'].sum().reset_index()
                    cat_monthly['Tanggal'] = cat_monthly['Tanggal'].dt.to_timestamp()
                    fig_cat_trend = px.line(cat_monthly, x='Tanggal', y='Total',
                                            title=f"Tren Bulanan - {selected_detail_cat}",
                                            labels={'Total': 'Pengeluaran (Rp)'},
                                            markers=True)
                    st.plotly_chart(fig_cat_trend, use_container_width=True)
                
                # Stats for selected category
                st.markdown(f"**Statistik {selected_detail_cat}:**")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("Total Pengeluaran", f"Rp{cat_detail['Total'].sum():,.0f}")
                with col_s2:
                    st.metric("Jumlah Transaksi", f"{len(cat_detail):,}")
                with col_s3:
                    st.metric("Rata-rata per Transaksi", f"Rp{cat_detail['Total'].mean():,.0f}")
                with col_s4:
                    top_produk = cat_detail['Produk'].mode().iloc[0] if not cat_detail.empty else '-'
                    st.metric("Produk Terbanyak", top_produk)
    else:
        st.info("Tidak ada data pengeluaran untuk ditampilkan.")

# ---- TAB 3: A/B TESTING ----
with tab3:
    st.subheader("🧪 A/B Testing: Model A (GRU) vs Model B (Prophet + XGBoost)")
    
    if model_trained:
        ab_data = {
            'Metrik': ['MAE (Rp)', 'RMSE (Rp)', 'MAPE (%)', 'Waktu Training', 'Interpretabilitas'],
            'Model A — GRU/LSTM': ['N/A', 'N/A', 'N/A', 'Lambat (GPU)', 'Rendah'],
            'Model B — Prophet': [f"Rp{p_mae:,.0f}", f"Rp{p_rmse:,.0f}", f"{p_mape:.1f}%", 'Cepat', 'Tinggi'],
            'Model B — XGBoost': [f"Rp{xgb_mae:,.0f}", f"Rp{xgb_rmse:,.0f}", f"{xgb_mape:.1f}%", 'Sangat Cepat', 'Sedang'],
        }
        st.dataframe(pd.DataFrame(ab_data), use_container_width=True, hide_index=True)
        
        # Visual comparison
        test_dates_plot = daily_cashflow.index[split_idx:]
        actual_test = daily_cashflow['Net_Cashflow'].iloc[split_idx:].values
        
        fig_ab = go.Figure()
        fig_ab.add_trace(go.Scatter(x=test_dates_plot, y=actual_test,
                                    name='Aktual', line=dict(color='black', width=2)))
        fig_ab.add_trace(go.Scatter(x=test_dates_plot, y=prophet_test_pred,
                                    name='Prophet', line=dict(color='#FF5722', dash='dash')))
        fig_ab.add_trace(go.Scatter(x=test_dates_plot, y=y_pred_xgb,
                                    name='XGBoost', line=dict(color='#4CAF50', dash='dot')))
        fig_ab.update_layout(title="Perbandingan Model pada Data Test",
                             xaxis_title="Tanggal", yaxis_title="Net Cashflow (Rp)", height=400)
        st.plotly_chart(fig_ab, use_container_width=True)
    else:
        st.info("Model belum dilatih. Tambahkan lebih banyak data untuk training model.")

# ---- TAB 4: TREN & POLA ----
with tab4:
    st.subheader("📅 Tren & Pola Pengeluaran")
    
    if not expense_df.empty:
        # Day pattern
        expense_df['Hari'] = expense_df['Tanggal'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_expense = expense_df.groupby('Hari')['Total'].mean().reindex(day_order)
        
        fig_daily = px.bar(x=day_expense.index, y=day_expense.values,
                           title="Rata-rata Pengeluaran per Hari",
                           labels={'x': 'Hari', 'y': 'Rata-rata Pengeluaran (Rp)'},
                           color=day_expense.values, color_continuous_scale='Blues')
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Monthly pattern
        expense_df['Bulan_Num'] = expense_df['Tanggal'].dt.month
        monthly_pattern = expense_df.groupby('Bulan_Num')['Total'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']
        
        fig_monthly_pattern = px.line(x=month_names, y=monthly_pattern.values,
                                      title="Rata-rata Pengeluaran per Bulan (Seasonality)",
                                      labels={'x': 'Bulan', 'y': 'Rata-rata Pengeluaran (Rp)'},
                                      markers=True)
        fig_monthly_pattern.add_hline(y=monthly_pattern.mean(), line_dash='dash', line_color='red',
                                      annotation_text="Rata-rata")
        st.plotly_chart(fig_monthly_pattern, use_container_width=True)
        
        # Heatmap
        heatmap_data = expense_df.groupby(['Hari', 'Bulan_Num'])['Total'].mean().unstack()
        heatmap_data = heatmap_data.reindex(day_order)
        
        fig_heatmap = px.imshow(heatmap_data,
                                title="Heatmap Pengeluaran (Hari vs Bulan)",
                                labels=dict(x="Bulan", y="Hari", color="Rata-rata (Rp)"),
                                color_continuous_scale='Reds', aspect='auto')
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Tidak ada data pengeluaran untuk analisis tren.")

# ---- TAB 5: RINGKASAN ----
with tab5:
    st.subheader("📋 Ringkasan Keuangan GenZ")
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown("### Statistik Pengeluaran")
        if not expense_df.empty:
            st.metric("Total Pengeluaran", f"Rp{expense_df['Total'].sum():,.0f}")
            st.metric("Rata-rata per Hari", f"Rp{expense_df.groupby(expense_df['Tanggal'].dt.date)['Total'].sum().mean():,.0f}")
            st.metric("Rata-rata per Transaksi", f"Rp{expense_df['Total'].mean():,.0f}")
            st.metric("Total Transaksi", f"{len(expense_df):,}")
    
    with col_r2:
        st.markdown("### Statistik Pemasukan")
        income_data = df[df['Tipe_Transaksi'] == 'Pemasukan']
        if not income_data.empty:
            st.metric("Total Pemasukan", f"Rp{income_data['Total'].sum():,.0f}")
            st.metric("Rata-rata per Bulan", f"Rp{income_data.groupby(income_data['Tanggal'].dt.to_period('M'))['Total'].sum().mean():,.0f}")
            st.metric("Sumber Terbanyak", f"{income_data['Sumber'].mode().iloc[0] if not income_data.empty else '-'}")
            st.metric("Total Transaksi Pemasukan", f"{len(income_data):,}")
        else:
            st.info("Tidak ada data pemasukan.")
    
    st.markdown("---")
    st.markdown("### 💡 Insight & Rekomendasi")
    
    # Generate insights
    insights = []
    
    if not category_total.empty:
        top3 = category_total.head(3)['Kategori'].tolist()
        insights.append(f"🔴 **Top 3 Kategori Pengeluaran:** {', '.join(top3)}")
    
    if net_cashflow < 0:
        insights.append("⚠️ **Net Cashflow Negatif:** Pengeluaran melebihi pemasukan. Perlu mengurangi pengeluaran atau menambah sumber pemasukan.")
    else:
        insights.append(f"✅ **Net Cashflow Positif:** Rp{net_cashflow/1e6:.1f}Jt surplus. Pertahankan kebiasaan baik ini!")