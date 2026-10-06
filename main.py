import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression


# =========================================================
# 1. 기본 설정
# =========================================================

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연평균 기온 데이터를 이용하여 "
    "장기간의 기온 변화 추세를 분석하고 미래 기온을 예측합니다."
)


# =========================================================
# 2. 데이터 불러오기
# =========================================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

df = pd.read_csv(
    DATA_URL,
    encoding="utf-8"
)


# =========================================================
# 3. 데이터 전처리
# =========================================================

df["날짜"] = pd.to_datetime(
    df["날짜"],
    errors="coerce"
)

df["평균기온"] = pd.to_numeric(
    df["평균기온"],
    errors="coerce"
)

df = df.dropna(
    subset=["날짜", "평균기온"]
)

# 2025년까지의 데이터만 사용
df = df[
    df["날짜"].dt.year <= 2025
].copy()

# 연도 추출
df["연도"] = df["날짜"].dt.year


# =========================================================
# 4. 1년에 300일 이상 관측된 연도만 사용
# =========================================================

year_count = (
    df.groupby("연도")
    .size()
)

valid_years = year_count[
    year_count >= 300
].index

df = df[
    df["연도"].isin(valid_years)
].copy()


# =========================================================
# 5. 연도별 평균기온 계산
# =========================================================

yearly = (
    df.groupby("연도")["평균기온"]
    .mean()
    .reset_index()
)

yearly.columns = [
    "연도",
    "연평균기온"
]

yearly = yearly.sort_values(
    "연도"
).reset_index(drop=True)


# 1908년을 기준으로 경과 연수 계산
yearly["경과연수"] = (
    yearly["연도"] - 1908
)


# =========================================================
# 6. 전체 기간 선형회귀
# =========================================================

X_all = yearly[["경과연수"]]
y_all = yearly["연평균기온"]

model_all = LinearRegression()

model_all.fit(
    X_all,
    y_all
)

yearly["전체기간_예측"] = model_all.predict(
    X_all
)

# 전체 기간 기울기
slope_all = model_all.coef_[0]

# 100년당 변화량
slope_all_100 = slope_all * 100


# =========================================================
# 7. 최근 20년 데이터
# =========================================================

recent_20 = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()


# 최근 20년 회귀모델
X_recent = recent_20[["경과연수"]]
y_recent = recent_20["연평균기온"]

model_recent = LinearRegression()

model_recent.fit(
    X_recent,
    y_recent
)

recent_20["최근20년_예측"] = model_recent.predict(
    X_recent
)

# 최근 20년 기울기
slope_recent = model_recent.coef_[0]

# 100년당 변화량
slope_recent_100 = slope_recent * 100


# =========================================================
# 8. 데이터 기본 정보
# =========================================================

st.subheader("📊 분석에 사용한 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "사용한 연도 수",
        f"{len(yearly)}년"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{yearly['연도'].min()}년"
    )

with col3:
    st.metric(
        "마지막 연도",
        f"{yearly['연도'].max()}년"
    )


# =========================================================
# 9. 전체 기간 기울기 크게 표시
# =========================================================

st.subheader("🌡️ 직선의 기울기")

st.metric(
    "전체 기간 : 100년에 기온이 얼마나 변하는가?",
    f"{slope_all_100:.2f} °C / 100년"
)

st.write(
    f"전체 기간 회귀선의 1년당 기울기: "
    f"{slope_all:.4f} °C"
)


# =========================================================
# 10. 전체 기간 vs 최근 20년 비교
# =========================================================

st.subheader(
    "📈 전체 기간과 최근 20년의 기울기 비교"
)

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 전체 기간")

    st.metric(
        "100년에 몇 도 변하는가?",
        f"{slope_all_100:.2f} °C"
    )

    st.write(
        f"분석 기간: "
        f"{yearly['연도'].min()}~{yearly['연도'].max()}"
    )

with col2:

    st.markdown("### 최근 20년")

    st.metric(
        "100년에 몇 도 변하는가?",
        f"{slope_recent_100:.2f} °C"
    )

    st.write(
        "분석 기간: 2006~2025"
    )


# =========================================================
# 11. 기울기 차이
# =========================================================

difference = (
    slope_recent_100 - slope_all_100
)

st.write("")

if difference > 0:

    st.info(
        f"최근 20년의 기온 상승 속도가 "
        f"전체 기간보다 100년 기준 "
        f"{difference:.2f} °C 더 큽니다."
    )

elif difference < 0:

    st.info(
        f"최근 20년의 기온 상승 속도가 "
        f"전체 기간보다 100년 기준 "
        f"{abs(difference):.2f} °C 더 작습니다."
    )

else:

    st.info(
        "최근 20년과 전체 기간의 기울기가 같습니다."
    )


# =========================================================
# 12. 전체 기간 산점도 + 회귀선
# =========================================================

st.subheader(
    "📉 서울 연평균 기온 변화와 전체 기간 회귀선"
)

fig = go.Figure()


# 실제 연평균 기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균 기온",
        hovertemplate=(
            "연도: %{x}<br>"
            "연평균 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)


# 전체 기간 회귀선
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["전체기간_예측"],
        mode="lines",
        name="전체 기간 회귀선",
        hovertemplate=(
            "연도: %{x}<br>"
            "회귀선: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (°C)",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 13. 최근 20년 회귀선 비교 그래프
# =========================================================

st.subheader(
    "🔍 최근 20년 실제 기온과 회귀선"
)

fig_recent = go.Figure()


fig_recent.add_trace(
    go.Scatter(
        x=recent_20["연도"],
        y=recent_20["연평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        hovertemplate=(
            "연도: %{x}<br>"
            "연평균 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)


fig_recent.add_trace(
    go.Scatter(
        x=recent_20["연도"],
        y=recent_20["최근20년_예측"],
        mode="lines",
        name="최근 20년 회귀선",
        hovertemplate=(
            "연도: %{x}<br>"
            "회귀선: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)


fig_recent.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (°C)",
    hovermode="x unified"
)

st.plotly_chart(
    fig_recent,
    use_container_width=True
)


# =========================================================
# 14. 연도별 데이터
# =========================================================

st.subheader("📋 연도별 연평균 기온")

display_yearly = yearly[
    [
        "연도",
        "연평균기온"
    ]
].copy()

display_yearly["연평균기온"] = (
    display_yearly["연평균기온"]
    .round(2)
)

st.dataframe(
    display_yearly,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 15. 미래 연도 예측
# =========================================================

st.subheader("🔮 미래 연도 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

selected_elapsed = selected_year - 1908

predicted_temperature = model_all.predict(
    np.array([[selected_elapsed]])
)[0]


st.metric(
    f"{selected_year}년 예상 연평균 기온",
    f"{predicted_temperature:.2f} °C"
)


# =========================================================
# 16. 예측 계산 설명
# =========================================================

st.caption(
    "예측값은 전체 기간의 연평균 기온 데이터를 이용한 "
    "선형회귀 모델을 기준으로 계산했습니다."
)
