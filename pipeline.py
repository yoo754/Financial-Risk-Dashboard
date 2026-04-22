import pandas as pd
import numpy as np
import requests
import joblib
import os
import streamlit as st
from datetime import datetime, timedelta
from pykrx import stock as pykrx_stock
from dotenv import load_dotenv

load_dotenv()

ECOS_API_KEY = os.getenv("ECOS_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

STOCK_COLS = ["삼성전자", "SK하이닉스", "NAVER", "KODEX200", "KODEX채권"]
WEIGHTS    = np.array([0.25, 0.20, 0.20, 0.25, 0.10])

TICKERS_KRX = {
    "삼성전자":  "005930",
    "SK하이닉스": "000660",
    "NAVER":     "035420",
    "KODEX200":  "069500",
    "KODEX채권": "114820",
}

# 1. 주가 수집 
@st.cache_data(ttl=3600)
def fetch_realtime_stock(days: int = 60):
    end       = datetime.today()
    start     = end - timedelta(days=days)
    start_str = start.strftime("%Y%m%d")
    end_str   = end.strftime("%Y%m%d")

    price_dict = {}
    for name, ticker in TICKERS_KRX.items():
        try:
            df = pykrx_stock.get_market_ohlcv_by_date(start_str, end_str, ticker)
            if not df.empty:
                price_dict[name] = df["종가"]
        except Exception as e:
            st.warning(f"{name} 수집 실패: {e}")

    return pd.DataFrame(price_dict).ffill().dropna()

# 2. ECOS 금리 수집
@st.cache_data(ttl=3600)
def fetch_realtime_rates(days: int = 60):
    end   = datetime.today()
    start = end - timedelta(days=days)

    def _fetch(stat_code, item_code, name, freq="D"):
        if freq == "D":
            s = start.strftime("%Y%m%d")
            e = end.strftime("%Y%m%d")
            fmt = "%Y%m%d"
        else:
            s = start.strftime("%Y%m")
            e = end.strftime("%Y%m")
            fmt = "%Y%m"
        url = (
            f"https://ecos.bok.or.kr/api/StatisticSearch"
            f"/{ECOS_API_KEY}/json/kr/1/1000"
            f"/{stat_code}/{freq}/{s}/{e}/{item_code}"
        )
        resp = requests.get(url, timeout=10)
        rows = resp.json().get("StatisticSearch", {}).get("row", [])
        if not rows:
            return pd.Series(dtype=float, name=name)
        return pd.Series(
            [float(r["DATA_VALUE"]) for r in rows],
            index=pd.to_datetime([r["TIME"] for r in rows], format=fmt),
            name=name
        )

    base_rate = _fetch("722Y001", "0101000", "기준금리", freq="M")
    bond_3y   = _fetch("817Y002", "010200000", "국고채3년", freq="D")

    idx = pd.date_range(base_rate.index.min(), end, freq="D")
    base_rate = base_rate.reindex(idx).ffill()

    idx2 = pd.date_range(bond_3y.index.min(), end, freq="D")
    bond_3y = bond_3y.reindex(idx2).ffill()


    return pd.concat([base_rate, bond_3y], axis=1).ffill()

# 3. VIX 수집 
@st.cache_data(ttl=3600)
def fetch_realtime_vix(days: int = 60):
    end   = datetime.today()
    start = end - timedelta(days=days)
    url   = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id=VIXCLS"
        f"&observation_start={start.strftime('%Y-%m-%d')}"
        f"&observation_end={end.strftime('%Y-%m-%d')}"
        f"&api_key={FRED_API_KEY}&file_type=json"
    )
    resp = requests.get(url, timeout=10)
    obs  = resp.json().get("observations", [])
    vix  = pd.Series(
        {o["date"]: float(o["value"]) for o in obs if o["value"] != "."},
        name="VIX"
    )
    vix.index = pd.to_datetime(vix.index)
    return vix.ffill()

# 4. 피처 엔지니어링 
def preprocess(price_df, rate_df, vix_series):
    returns    = price_df.pct_change().dropna()
    port_ret   = (returns[STOCK_COLS] * WEIGHTS).sum(axis=1)

    macro_df   = pd.concat([rate_df, vix_series], axis=1)
    macro_df   = macro_df.reindex(returns.index).ffill().bfill()
    df         = pd.concat([returns, macro_df], axis=1)
    df["portfolio_return"] = port_ret

    for col in STOCK_COLS:
        df[f"{col}_vol5"]  = returns[col].rolling(5).std()
        df[f"{col}_vol20"] = returns[col].rolling(20).std()
        df[f"{col}_mom5"]  = returns[col].rolling(5).sum()

    df["port_vol5"]   = port_ret.rolling(5).std()
    df["port_vol20"]  = port_ret.rolling(20).std()
    df["port_mom5"]   = port_ret.rolling(5).sum()
    df["port_skew20"] = port_ret.rolling(20).skew()
    df["port_kurt20"] = port_ret.rolling(20).kurt()

    df["rate_spread"] = df["국고채3년"] - df["기준금리"]
    df["vix_change"]  = df["VIX"].pct_change()
    df["vix_ma5"]     = df["VIX"].rolling(5).mean()

    return df.dropna()

# 5. 모델 로드 + 예측 
@st.cache_resource
def load_model():
    return joblib.load("data/model.pkl")

FEATURE_COLS = [
    '삼성전자', 'SK하이닉스', 'NAVER', 'KODEX200', 'KODEX채권',
    '기준금리', '국고채3년', 'VIX',
    '삼성전자_vol5', '삼성전자_vol20', '삼성전자_mom5',
    'SK하이닉스_vol5', 'SK하이닉스_vol20', 'SK하이닉스_mom5',
    'NAVER_vol5', 'NAVER_vol20', 'NAVER_mom5',
    'KODEX200_vol5', 'KODEX200_vol20', 'KODEX200_mom5',
    'KODEX채권_vol5', 'KODEX채권_vol20', 'KODEX채권_mom5',
    'port_vol5', 'port_vol20', 'port_mom5',
    'port_skew20', 'port_kurt20',
    'rate_spread', 'vix_change', 'vix_ma5'
]

def predict_risk(feature_df):
    model  = load_model()
    latest = feature_df[FEATURE_COLS].iloc[[-1]]
    prob   = model.predict_proba(latest)[0][1]
    return {
        "prob":     prob,
        "pred":     int(prob >= 0.05),
        "features": latest.iloc[0].to_dict()
    }