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
    "서울의 연평균 기온 변화를 분석하고 "
    "선형회귀를 이용해 미래 기온을 예측합니다."
)


# =========================================================
# 2. 데이터 불러오기
# =========================================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


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

# 2025년까지만 사용
df = df[
    df["날짜"].dt.year <= 2025
].copy()

# 연도 만들기
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
# 5. 연도별 연평균 기온 계산
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


# 1908년을 기준으로 몇 년이 지났는지 계산
yearly["경과연수"] = (
    yearly["연도"] - 1908
)


# 데이터가 너무 적은 경우
if len(yearly) < 2:
    st.error(
        "회귀분석을 하기 위한 연도별 데이터가 충분하지 않습니다."
    )
    st.stop()


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

yearly["전체기간_예측"] = (
    model_all.predict(X_all)
)

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


if len(recent_20) >= 2:

    X_recent = recent_20[["경과연수"]]
    y_recent = recent_20["연평균기온"]

    model_recent = LinearRegression()

    model_recent.fit(
        X_recent,
        y_recent
    )

    recent_20["최근20년_예측"] = (
        model_recent.predict(X_recent)
    )

    slope_recent = model_recent.coef_[0]

    # 100년당 변화량
    slope_recent_100 = slope_recent * 100

else:
    model_recent = None
    slope_recent = np.nan
    slope_recent_100 = np.nan


# =========================================================
# 8. 기본 데이터 정보
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
        f"{int(yearly['연도'].min())}년"
    )

with col3:
    st.metric(
        "마지막 연도",
        f"{int(yearly['연도'].max())}년"
    )


# =========================================================
# 9. 핵심 결과
# =========================================================

st.subheader("🌡️ 기온 변화 속도")

st.write(
    "선형회귀선의 기울기를 100년 기준으로 환산했습니다."
)


col1, col2 = st.columns(2)

with col1:

    st.metric(
        label="전체 기간",
        value=f"{slope_all_100:+.2f} °C / 100년"
    )

    if slope_all_100 > 0:
        st.caption(
            f"전체 기간 동안 100년에 약 "
            f"{slope_all_100:.2f}°C 상승하는 추세입니다."
        )
    elif slope_all_100 < 0:
        st.caption(
            f"전체 기간 동안 100년에 약 "
            f"{abs(slope_all_100):.2f}°C 하락하는 추세입니다."
        )
    else:
        st.caption(
            "전체 기간의 선형적인 기온 변화가 거의 없습니다."
        )


with col2:

    if not np.isnan(slope_recent_100):

        st.metric(
            label="최근 20년 (2006~2025)",
            value=f"{slope_recent_100:+.2f} °C / 100년"
        )

        if slope_recent_100 > 0:
            st.caption(
                f"최근 20년의 추세를 100년으로 환산하면 "
                f"약 {slope_recent_100:.2f}°C 상승입니다."
            )
        elif slope_recent_100 < 0:
            st.caption(
                f"최근 20년의 추세를 100년으로 환산하면 "
                f"약 {abs(slope_recent_100):.2f}°C 하락입니다."
            )
        else:
            st.caption(
                "최근 20년의 선형적인 기온 변화가 거의 없습니다."
            )

    else:
        st.metric(
            label="최근 20년 (2006~2025)",
            value="계산 불가"
        )


# =========================================================
# 10. 기울기 비교
# =========================================================

st.subheader("🔎 전체 기간과 최근 20년 기울기 비교")

comparison = pd.DataFrame({
    "구간": [
        "전체 기간",
        "최근 20년 (2006~2025)"
    ],
    "연간 기울기 (°C)": [
        slope_all,
        slope_recent
    ],
    "100년당 기울기 (°C)": [
        slope_all_100,
        slope_recent_100
    ]
})

comparison["연간 기울기 (°C)"] = (
    comparison["연간 기울기 (°C)"].round(4)
)

comparison["100년당 기울기 (°C)"] = (
    comparison["100년당 기울기 (°C)"].round(2)
)

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 11. 연평균 기온 + 전체 기간 회귀선 그래프
# =========================================================

st.subheader("📈 서울 연평균 기온 변화와 전체 기간 회귀선")

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
            "연평균 기온: %{y:.2f}°C"
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
            "회귀선: %{y:.2f}°C"
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
# 12. 최근 20년 회귀선 비교 그래프
# =========================================================

if model_recent is not None:

    st.subheader(
        "📈 최근 20년 실제 기온과 회귀선"
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
                "연평균 기온: %{y:.2f}°C"
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
                "회귀선: %{y:.2f}°C"
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
# 13. 연도 선택 및 예측
# =========================================================

st.subheader("🔮 연도별 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
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
    label=f"{selected_year}년 예상 연평균 기온",
    value=f"{predicted_temperature:.2f} °C"
)


# =========================================================
# 14. 회귀식 정보
# =========================================================

st.subheader("📐 회귀모델 정보")

st.write(
    f"전체 기간 회귀식의 연간 기울기: "
    f"**{slope_all:.4f} °C/년**"
)

st.write(
    f"100년당 기온 변화: "
    f"**{slope_all_100:+.2f} °C/100년**"
)

if model_recent is not None:

    st.write(
        f"최근 20년 회귀식의 연간 기울기: "
        f"**{slope_recent:.4f} °C/년**"
    )

    st.write(
        f"최근 20년 회귀선을 100년 기준으로 환산하면: "
        f"**{slope_recent_100:+.2f} °C/100년**"
    )
