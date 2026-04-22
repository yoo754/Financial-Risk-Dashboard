import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import urllib.request
import os
# import lightgbm as lgb
# from scipy import stats
from pipeline import (fetch_realtime_stock, fetch_realtime_rates,
                      fetch_realtime_vix, preprocess, predict_risk)

@st.cache_resource
def set_korean_font():
    font_url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
    font_path = "/tmp/NanumGothic.ttf"
    if not os.path.exists(font_path):
        urllib.request.urlretrieve(font_url, font_path)
    fm.fontManager.addfont(font_path)
    matplotlib.rcParams["font.family"] = "NanumGothic"
    matplotlib.rcParams["axes.unicode_minus"] = False

set_korean_font()

st.set_page_config(
    page_title="금융상품 리스크 분석 시스템",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    master_df = pd.read_csv("data/master_dataset.csv",  index_col=0, parse_dates=True)
    metrics_s = pd.read_csv("data/financial_metrics.csv", index_col=0).squeeze()
    pred_df   = pd.read_csv("data/ml_predictions.csv",  index_col=0, parse_dates=True)
    return master_df, metrics_s, pred_df

master_df, metrics_s, pred_df = load_data()

STOCK_COLS = ["삼성전자", "SK하이닉스", "NAVER", "KODEX200", "KODEX채권"]
WEIGHTS    = np.array([0.25, 0.20, 0.20, 0.25, 0.10])


# 사이드바 — 투자금액만
# ════════════════════════════════════════════════════════════════
st.sidebar.title("⚙️ 리스크 파라미터 설정")
investment = st.sidebar.number_input(
    "💰 투자금액 (원)", min_value=1_000_000, max_value=10_000_000_000,
    value=100_000_000, step=10_000_000, format="%d"
)
st.sidebar.info("각 탭 상단에서 해당 탭의 파라미터를 설정하세요.")

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ 실시간 리스크 예측")

if st.sidebar.button("🔄 최신 데이터로 예측하기", type="primary", use_container_width=True):
    with st.spinner("실시간 데이터 수집 중..."):
        try:
            price_df   = fetch_realtime_stock(days=60)
            rate_df    = fetch_realtime_rates(days=60)
            vix_series = fetch_realtime_vix(days=60)
            feature_df = preprocess(price_df, rate_df, vix_series)
            result     = predict_risk(feature_df)

            prob = result["prob"]

            if prob < 0.05:
                risk_label  = "🟢 저위험"
                delta_color = "normal"
            elif prob < 0.10:
                risk_label  = "🟡 주의"
                delta_color = "off"
            else:
                risk_label  = "🔴 고위험"
                delta_color = "inverse"

            st.sidebar.metric(
                label="실시간 VaR 초과 확률",
                value=f"{prob*100:.2f}%",
                delta=risk_label,
                delta_color=delta_color
            )
            st.sidebar.caption("수집 기준: 최근 60일 | 캐시: 1시간")
            st.sidebar.caption("※ 기본 비중(삼성전자 25% 등) 기준 예측")

            with st.sidebar.expander("📊 피처값 확인"):
                feats = result["features"]
                feat_display = pd.DataFrame(feats, index=["현재값"]).T
                feat_display.columns = ["현재값"]
                st.dataframe(feat_display.style.format("{:.6f}"),
                             use_container_width=True)

        except Exception as e:
            import traceback
            st.error(f"에러: {e}")
            st.code(traceback.format_exc())

st.sidebar.info("""
**임계값 0.05 기준**
- 🟢 5% 미만 → 안정적
- 🟡 5~10% → 주의
- 🔴 10% 이상 → 위험
""")

# 메인 타이틀
# ════════════════════════════════════════════════════════════════
st.title("📊 금융상품 리스크 분석 시스템")
st.markdown(f"**분석 기간:** {master_df.index[0].date()} ~ {master_df.index[-1].date()} &nbsp;|&nbsp; **거래일:** {len(master_df)}일")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📈 포트폴리오 개요", "⚠️ VaR 리스크", "🤖 ML 리스크 예측", "🏦 채권 분석"])

# TAB 1: 포트폴리오 개요 
with tab1:
    st.subheader("포트폴리오 현황")

    # 비중 슬라이더 — 탭 안에서
    with st.expander("⚙️ 포트폴리오 비중 설정", expanded=True):
        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        w1 = col_s1.slider("삼성전자",   0, 100, 25, key="w1")
        w2 = col_s2.slider("SK하이닉스", 0, 100, 20, key="w2")
        w3 = col_s3.slider("NAVER",     0, 100, 20, key="w3")
        w4 = col_s4.slider("KODEX200",  0, 100, 25, key="w4")
        w5 = col_s5.slider("KODEX채권", 0, 100, 10, key="w5")

    total_w = w1 + w2 + w3 + w4 + w5
    if total_w == 0:
        weights = WEIGHTS
    else:
        weights = np.array([w1, w2, w3, w4, w5]) / total_w

    if total_w != 100:
        st.warning(f"⚠️ 비중 합계: {total_w}% (자동 정규화 적용)")

    port_ret = (master_df[STOCK_COLS] * weights).sum(axis=1)

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    cum_total = (1 + port_ret).prod() - 1
    ann_vol   = port_ret.std() * np.sqrt(252)
    sharpe    = (port_ret.mean() * 252) / (port_ret.std() * np.sqrt(252))

    c1.metric("📈 누적 수익률",  f"{cum_total*100:.2f}%")
    c2.metric("📉 연간 변동성",  f"{ann_vol*100:.2f}%")
    c3.metric("⚡ 샤프 비율",    f"{sharpe:.3f}")
    c4.metric("💰 투자금액",     f"{investment/1_000_000:.0f}백만원")

    fig, ax = plt.subplots(figsize=(12, 4))
    cum_ret = (1 + port_ret).cumprod() - 1
    ax.fill_between(cum_ret.index, cum_ret*100, 0,
                    where=cum_ret >= 0, color="#2196F3", alpha=0.3)
    ax.fill_between(cum_ret.index, cum_ret*100, 0,
                    where=cum_ret <  0, color="#F44336", alpha=0.3)
    ax.plot(cum_ret.index, cum_ret*100, color="#1565C0", linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("포트폴리오 누적 수익률")
    ax.set_ylabel("수익률 (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)
    with col1:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.pie(weights * 100, labels=STOCK_COLS, autopct="%1.1f%%",
                colors=["#42A5F5","#66BB6A","#FFA726","#AB47BC","#26A69A"])
        ax2.set_title("포트폴리오 비중")
        st.pyplot(fig2)
        plt.close()

    with col2:
        st.subheader("종목별 기술통계")
        stats_df = master_df[STOCK_COLS].describe().T[["mean","std","min","max"]]
        stats_df.columns = ["평균수익률", "변동성", "최소", "최대"]
        stats_df = stats_df.applymap(lambda x: f"{x*100:.3f}%")
        st.dataframe(stats_df, use_container_width=True)

# TAB 2: VaR 리스크 
with tab2:
    st.subheader("⚠️ Value at Risk (VaR) 분석")

    # 신뢰수준/보유기간 슬라이더 — 탭 안에서
    with st.expander("⚙️ VaR 파라미터 설정", expanded=True):
        col_v1, col_v2 = st.columns(2)
        confidence = col_v1.slider("📊 신뢰수준 (%)", 90, 99, 95, key="conf") / 100
        holding    = col_v2.slider("📅 보유기간 (일)", 1, 30, 10, key="hold")

    # tab2에서 쓸 port_ret (기본 WEIGHTS 기반)
    port_ret_var  = (master_df[STOCK_COLS] * WEIGHTS).sum(axis=1)
    var_threshold = np.percentile(port_ret_var, (1 - confidence) * 100)

    hist_var_1d = abs(np.percentile(port_ret_var, (1 - confidence) * 100)) * investment
    hist_var_nd = hist_var_1d * np.sqrt(holding)
    mu, sigma   = port_ret_var.mean(), port_ret_var.std()
    np.random.seed(42)
    sim         = np.random.normal(mu * holding, sigma * np.sqrt(holding), 100_000)
    mc_var      = abs(np.percentile(sim, (1 - confidence) * 100)) * investment
    tail        = port_ret_var[port_ret_var <= var_threshold]
    cvar        = abs(tail.mean()) * investment

    c1, c2, c3 = st.columns(3)
    c1.metric(f"📊 히스토리컬 VaR ({holding}일)",
              f"{hist_var_nd/1_000_000:.2f}백만원",
              delta=f"1일: {hist_var_1d/1_000_000:.2f}M")
    c2.metric(f"🎲 몬테카를로 VaR ({holding}일)",
              f"{mc_var/1_000_000:.2f}백만원")
    c3.metric("🔴 CVaR (Expected Shortfall)",
              f"{cvar/1_000_000:.2f}백만원",
              delta="최악 꼬리 리스크", delta_color="inverse")

    var_ratio  = hist_var_nd / investment * 100
    cvar_ratio = cvar / investment * 100
    mc_diff    = abs(mc_var - hist_var_nd) / hist_var_nd * 100

    if var_ratio < 2:
        var_level   = "🟢 안전"
        var_comment = f"현재 {holding}일 VaR는 투자금액의 **{var_ratio:.1f}%** 수준으로 매우 안정적입니다."
    elif var_ratio < 5:
        var_level   = "🟡 주의"
        var_comment = f"현재 {holding}일 VaR는 투자금액의 **{var_ratio:.1f}%** 수준입니다. 시장 변동에 주의가 필요합니다."
    else:
        var_level   = "🔴 위험"
        var_comment = f"현재 {holding}일 VaR는 투자금액의 **{var_ratio:.1f}%** 수준으로 손실 위험이 높습니다."

    tail_comment = (f"CVaR({cvar_ratio:.1f}%)가 VaR와 큰 차이 없어 극단적 손실 가능성은 낮습니다."
                    if cvar_ratio < var_ratio * 1.5
                    else f"CVaR({cvar_ratio:.1f}%)가 VaR보다 크게 높아 꼬리 리스크에 주의가 필요합니다.")
    mc_comment  = (f"히스토리컬 VaR와 몬테카를로 VaR의 차이가 {mc_diff:.1f}%로 두 방법론이 일치합니다."
                   if mc_diff < 10
                   else f"히스토리컬 VaR와 몬테카를로 VaR의 차이가 {mc_diff:.1f}%로, 수익률 분포가 정규분포와 다를 수 있습니다.")

    st.markdown(f"""
**리스크 수준: {var_level}**　　{var_comment}  
{tail_comment}  
{mc_comment}
    """)

    with st.expander("📖 지표 개념 이해하기"):
        st.markdown(f"""
**📊 히스토리컬 VaR** — *과거 데이터 기반*  
과거 실제 수익률을 그대로 사용해 손실 분포를 만듭니다.  
→ **"과거에 이 정도 손실이 {int((1-confidence)*100)}번 중 1번 있었다"** 는 방식입니다.  
현재 {holding}일 기준 **{hist_var_nd/1_000_000:.2f}백만원** 손실이 {int(confidence*100)}% 확률로 이 이하입니다.

---

**🎲 몬테카를로 VaR** — *수만 번 시뮬레이션 기반*  
과거 패턴을 바탕으로 **10만 번의 가상 시나리오**를 만들어 손실을 추정합니다.  
→ **"앞으로 일어날 수 있는 경우의 수를 엄청나게 많이 굴려서 평균 낸 것"** 입니다.  
히스토리컬 VaR와 차이가 {mc_diff:.1f}%로, {'두 방법이 비슷한 결과를 보입니다.' if mc_diff < 10 else '차이가 있어 수익률이 정규분포와 다를 수 있습니다.'}

---

**🔴 CVaR (Expected Shortfall)** — *VaR을 넘는 최악의 평균 손실*  
VaR은 "이 선을 넘을 확률이 {int((1-confidence)*100)}%"라는 **경계선**만 알려주지만,  
CVaR은 **그 경계를 넘었을 때 평균적으로 얼마나 잃는지**를 알려줍니다.  
→ 현재 CVaR **{cvar/1_000_000:.2f}백만원**은 최악 상황에서의 평균 손실입니다.
        """)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(port_ret_var*100, bins=60, color="#90CAF9", edgecolor="white",
                alpha=0.8, density=True)
        ax.axvline(var_threshold*100, color="#F44336", linewidth=2,
                   linestyle="--", label=f"VaR ({var_threshold*100:.2f}%)")
        ax.set_title("수익률 분포 & VaR 임계값")
        ax.set_xlabel("일간 수익률 (%)")
        ax.legend()
        st.pyplot(fig)
        plt.close()

        skew = port_ret_var.skew()
        kurt = port_ret_var.kurt()
        skew_comment = (f"왜도 {skew:.2f} → 손실 방향으로 치우친 분포입니다. 하락 리스크에 주의하세요." if skew < -0.5
                        else f"왜도 {skew:.2f} → 수익 방향으로 치우친 분포입니다." if skew > 0.5
                        else f"왜도 {skew:.2f} → 비교적 대칭적인 분포입니다.")
        kurt_comment = (f"첨도 {kurt:.2f} → 정규분포보다 꼬리가 두꺼워 극단적 손실 발생 가능성이 있습니다." if kurt > 1
                        else f"첨도 {kurt:.2f} → 정규분포에 가까운 형태입니다.")
        st.caption(f"📊 {skew_comment}  \n📊 {kurt_comment}")

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        rolling_var = port_ret_var.rolling(60).quantile(1 - confidence) * investment / 1_000_000
        ax.plot(rolling_var.index, rolling_var.abs(), color="#AB47BC", linewidth=1.2)
        ax.fill_between(rolling_var.index, rolling_var.abs(), alpha=0.2, color="#AB47BC")
        ax.set_title("60일 롤링 VaR 추이")
        ax.set_ylabel("VaR (백만원)")
        st.pyplot(fig)
        plt.close()

        rolling_clean = rolling_var.dropna()
        recent_var    = rolling_clean.iloc[-1]
        avg_var       = rolling_clean.mean()
        max_var       = rolling_clean.abs().max()

        trend_comment = (f"최근 VaR({abs(recent_var):.2f}백만원)가 평균({abs(avg_var):.2f}백만원)보다 높아 **리스크가 확대**되고 있습니다." if abs(recent_var) > abs(avg_var) * 1.2
                         else f"최근 VaR({abs(recent_var):.2f}백만원)가 평균({abs(avg_var):.2f}백만원)보다 낮아 **리스크가 안정**되고 있습니다." if abs(recent_var) < abs(avg_var) * 0.8
                         else f"최근 VaR({abs(recent_var):.2f}백만원)가 평균({abs(avg_var):.2f}백만원) 수준으로 **안정적으로 유지**되고 있습니다.")
        st.caption(f"📈 {trend_comment}  \n📈 분석 기간 중 최대 VaR는 **{max_var:.2f}백만원**이었습니다.")
        
        
# TAB 3: ML 리스크 예측 
with tab3:
    st.subheader("🤖 ML 리스크 예측 모델")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC", "0.7362", delta="+47% vs 랜덤")
    c2.metric("Recall",  "25.00%", delta="전통 VaR 0% 대비")
    c3.metric("Precision", "33.33%")
    c4.metric("최적 임계값", "0.05")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(pred_df.index, pred_df["ml_prob_risk"]*100,
                color="#42A5F5", linewidth=1.0, alpha=0.7, label="ML 리스크 확률")
        actual_risk = pred_df[pred_df["actual"] == 1]
        ax.scatter(actual_risk.index, actual_risk["ml_prob_risk"]*100,
                   color="#F44336", s=40, zorder=5, label="실제 VaR 초과")
        ax.axhline(5, color="#FF9800", linewidth=1.5, linestyle="--", label="임계값 5%")
        ax.set_title("ML 리스크 확률 (테스트 기간)")
        ax.set_ylabel("위험 확률 (%)")
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        fig.autofmt_xdate(rotation=30, ha="right")
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        cats  = ["Recall", "Precision", "AUC×10"]
        trad  = [0.00, 0.00, 5.00]
        ml    = [0.25, 0.333, 7.362]
        x     = np.arange(len(cats))
        ax.bar(x - 0.2, trad, 0.35, label="전통 VaR", color="#B0BEC5")
        ax.bar(x + 0.2, ml,   0.35, label="ML VaR",   color="#42A5F5")
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.set_title("전통 VaR vs ML VaR 성능")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    # ── SHAP 로드 ──
    @st.cache_data
    def load_shap():
        return pd.read_csv("data/shap_importance.csv")

    shap_df = load_shap()
    top1 = shap_df.iloc[0]["feature"]
    top2 = shap_df.iloc[1]["feature"]
    top3 = shap_df.iloc[2]["feature"]

    feature_kor = {
        "rolling_vol_5":   "최근 5일 변동성",
        "rolling_vol_20":  "최근 20일 변동성",
        "return_lag1":     "전일 수익률",
        "return_lag2":     "2일 전 수익률",
        "return_lag3":     "3일 전 수익률",
        "rolling_mean_5":  "5일 평균 수익률",
        "rolling_mean_20": "20일 평균 수익률",
        "var_hist":        "히스토리컬 VaR",
        "skewness":        "수익률 왜도",
        "kurtosis":        "수익률 첨도",
    }
    top1_kor = feature_kor.get(top1, top1)
    top2_kor = feature_kor.get(top2, top2)
    top3_kor = feature_kor.get(top3, top3)

    # ── 모델 자동 해석 텍스트 ──
    st.subheader("🧠 AI 모델 해석")

    st.image("data/shap_importance.png",
             caption="SHAP Feature Importance — 각 피처가 VaR 초과 리스크 예측에 미치는 영향",
             use_container_width=True)

    recent = pred_df.tail(20)
    recent_risk_avg = recent["ml_prob_risk"].mean() * 100
    recent_hit      = pred_df[pred_df["actual"] == 1]["ml_prob_risk"].mean() * 100
    high_risk_days  = (pred_df["ml_prob_risk"] > 0.05).sum()
    total_days      = len(pred_df)

    if recent_risk_avg < 3:
        risk_level   = "🟢 낮음"
        risk_comment = f"최근 20일 평균 위험 확률이 {recent_risk_avg:.1f}%로 안정적인 구간입니다."
    elif recent_risk_avg < 6:
        risk_level   = "🟡 보통"
        risk_comment = f"최근 20일 평균 위험 확률이 {recent_risk_avg:.1f}%로 임계값(5%) 근처입니다. 주의가 필요합니다."
    else:
        risk_level   = "🔴 높음"
        risk_comment = f"최근 20일 평균 위험 확률이 {recent_risk_avg:.1f}%로 임계값(5%)을 초과했습니다. 리스크 관리가 필요합니다."

    high_risk_ratio = high_risk_days / total_days * 100
    if high_risk_ratio < 5:
        detect_comment = f"전체 기간 중 {high_risk_ratio:.1f}%의 날만 위험 신호를 발생시켜 보수적으로 작동하고 있습니다."
    elif high_risk_ratio < 15:
        detect_comment = f"전체 기간 중 {high_risk_ratio:.1f}%의 날에 위험 신호를 발생시켜 적절한 민감도를 유지합니다."
    else:
        detect_comment = f"전체 기간 중 {high_risk_ratio:.1f}%의 날에 위험 신호를 발생시켜 다소 민감하게 반응하고 있습니다."

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
**현재 위험 수준: {risk_level}**  
{risk_comment}

**모델 동작 패턴**  
{detect_comment}

**📌 리스크 주요 원인 (SHAP)**  
이 모델이 리스크를 판단할 때 가장 크게 영향을 미치는 요인은  
**1위 {top1_kor}**, **2위 {top2_kor}**, **3위 {top3_kor}** 순입니다.
        """)
    with col2:
        st.markdown(f"""
**모델 성능 해석**  
AUC 0.74는 랜덤(0.5)보다 **47% 향상**된 수치로, 10번 중 약 7번은 리스크를 올바르게 판단합니다.

**실제 VaR 초과 구간 탐지**  
실제 손실이 VaR을 초과한 날의 평균 위험 확률은 **{recent_hit:.1f}%**로, 일반 구간보다 높게 나타납니다.
        """)

    # ── ML 예측 결과 상세 ──
    st.subheader("🔍 ML 예측 결과 상세")

    st.caption("""
**실제(VaR초과)**: 실제로 손실이 VaR 임계값을 넘었으면 1, 안 넘었으면 0  
**ML예측**: 모델이 위험하다고 판단하면 1, 아니면 0  
**오탐(False Positive)**: 실제=0인데 ML예측=1 → 모델이 과민 반응한 날  
**미탐(False Negative)**: 실제=1인데 ML예측=0 → 모델이 위험을 놓친 날
    """)

    display_df = pred_df[["actual", "ml_pred", "ml_prob_risk"]].copy()
    display_df.columns = ["실제(VaR초과)", "ML예측", "위험확률"]
    display_df["위험확률"] = display_df["위험확률"].map(lambda x: f"{x*100:.2f}%")

    def label_result(row):
        a = row["실제(VaR초과)"]
        p = row["ML예측"]
        if   a == 1 and p == 1: return "✅ 정탐"
        elif a == 0 and p == 0: return "⬜ 정상"
        elif a == 0 and p == 1: return "🟡 오탐(과민반응)"
        else:                   return "🔴 미탐(위험 놓침)"

    display_df["판정"] = display_df.apply(label_result, axis=1)
    st.dataframe(display_df.tail(30), use_container_width=True)



# ── TAB 4: 채권 분석 ────────────────────────────────────────
with tab4:
    st.subheader("🏦 Duration & Convexity 분석")

    # ── 개념 설명 expander ──
    with st.expander("📖 Duration & Convexity란?"):
        st.markdown("""
**📐 Macaulay Duration** — 실질 회수 기간  
채권에서 돈을 전부 회수하는 데 걸리는 **가중 평균 기간(년)**입니다.  
만기가 길수록, 쿠폰이 낮을수록 Duration이 길어집니다.

**📉 Modified Duration** — 금리 민감도  
금리가 **1%p 변할 때 채권 가격이 몇 % 변하는지**를 나타냅니다.  
→ Modified Duration이 3이면, 금리 1%p 상승 시 채권 가격 약 **3% 하락**

**〰️ Convexity (볼록성)** — Duration의 오차 보정  
금리 변화가 클수록 Duration만으로는 가격 변화를 정확히 추정하기 어렵습니다.  
Convexity가 클수록 금리 상승 시 손실이 줄고, 금리 하락 시 이익이 더 커집니다.  
→ **Convexity가 높은 채권 = 투자자에게 유리**
        """)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**채권 파라미터 설정**")
        face_value  = st.number_input("액면가 (원)", value=10_000, step=1_000)
        coupon_rate = st.slider("표면금리 (%)", 0.0, 10.0, 3.5) / 100
        ytm         = st.slider("만기수익률 YTM (%)", 0.1, 10.0,
                                float(master_df["국고채3년"].iloc[-1])) / 100
        periods     = st.slider("만기 (년)", 1, 10, 3)

        r          = ytm
        cash_flows = [face_value * coupon_rate] * periods
        cash_flows[-1] += face_value
        pv_list    = [cf / (1 + r) ** (t + 1) for t, cf in enumerate(cash_flows)]
        price      = sum(pv_list)
        weighted   = [(t + 1) * pv for t, pv in enumerate(pv_list)]
        mac_dur    = sum(weighted) / price
        mod_dur    = mac_dur / (1 + r)
        conv_num   = sum((t+1)*(t+2)*pv for t, pv in enumerate(pv_list))
        convexity  = conv_num / (price * (1 + r) ** 2)

        st.metric("Macaulay Duration", f"{mac_dur:.4f}년")
        st.metric("Modified Duration", f"{mod_dur:.4f}")
        st.metric("Convexity",         f"{convexity:.4f}")

        delta_r   = st.slider("금리 변화 시뮬레이션 (%p)", -3.0, 3.0, 1.0) / 100
        price_chg = (-mod_dur * delta_r + 0.5 * convexity * delta_r**2) * 100
        st.metric(f"금리 {delta_r*100:+.1f}%p 시 가격변화",
                  f"{price_chg:+.4f}%",
                  delta_color="normal" if price_chg >= 0 else "inverse")

        # ── Duration 수준 자동 해석 ──
        st.markdown("---")
        if mod_dur < 1:
            dur_comment = "금리 변화에 매우 둔감한 **단기 채권**입니다. 금리 리스크가 낮습니다."
        elif mod_dur < 3:
            dur_comment = "금리 변화에 보통 수준으로 반응하는 **중기 채권**입니다."
        elif mod_dur < 6:
            dur_comment = "금리 변화에 민감한 **장기 채권**입니다. 금리 상승 시 주의가 필요합니다."
        else:
            dur_comment = "금리 변화에 매우 민감한 **초장기 채권**입니다. 금리 리스크가 높습니다."

        if coupon_rate < ytm:
            price_comment = "표면금리 < YTM → 채권이 **할인 발행**된 상태입니다 (가격 < 액면가)."
        elif coupon_rate > ytm:
            price_comment = "표면금리 > YTM → 채권이 **할증 발행**된 상태입니다 (가격 > 액면가)."
        else:
            price_comment = "표면금리 = YTM → 채권이 **액면가 발행**된 상태입니다."

        if delta_r > 0:
            sim_comment = f"금리 {delta_r*100:+.1f}%p 상승 시 채권 가격 **{price_chg:.2f}%** 하락 예상."
        elif delta_r < 0:
            sim_comment = f"금리 {delta_r*100:+.1f}%p 하락 시 채권 가격 **{abs(price_chg):.2f}%** 상승 예상."
        else:
            sim_comment = "금리 변화 없음 → 채권 가격 변동 없음."

        st.markdown(f"""
💡 {dur_comment}  
📌 {price_comment}  
🎯 {sim_comment}
        """)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        rate_changes  = np.linspace(-0.03, 0.03, 200)
        price_changes = (-mod_dur * rate_changes + 0.5 * convexity * rate_changes**2) * 100
        ax.plot(rate_changes*100, price_changes, color="#1565C0", linewidth=2.5)
        ax.fill_between(rate_changes*100, price_changes, 0,
                        where=price_changes >= 0, color="#66BB6A", alpha=0.2, label="가격 상승")
        ax.fill_between(rate_changes*100, price_changes, 0,
                        where=price_changes <  0, color="#EF5350", alpha=0.2, label="가격 하락")
        ax.axvline(delta_r*100, color="#FF9800", linewidth=2,
                   linestyle="--", label=f"시뮬레이션 {delta_r*100:+.1f}%p")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(f"금리 민감도 분석\nModified Duration: {mod_dur:.4f} | Convexity: {convexity:.4f}")
        ax.set_xlabel("금리 변화 (%p)")
        ax.set_ylabel("채권 가격 변화 (%)")
        ax.legend()
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.1f}%p"))
        st.pyplot(fig)
        plt.close()

        # ── 차트 해석 텍스트 ──
        max_gain = price_changes.max()
        max_loss = price_changes.min()
        st.caption(
            f"📊 금리 -3%p 시 최대 +{max_gain:.2f}% 상승 / 금리 +3%p 시 최대 {max_loss:.2f}% 하락  \n"
            f"〰️ 곡선이 아래로 볼록할수록 Convexity가 높아 투자자에게 유리한 구조입니다."
        )