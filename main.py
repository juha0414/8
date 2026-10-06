import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =========================================================
# 1. 기본 설정
# =========================================================

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write("서울의 연평균 기온 데이터를 이용해 연도별 기온 변화를 분석하고 미래 기온을 예측합니다.")


# =========================================================
# 2. 데이터 주소
# =========================================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# =========================================================
# 3. 데이터 불러오기
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜를 날짜형으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온을 숫자로 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 날짜 또는 평균기온이 없는 행 제거
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.code(str(e))
    st.stop()


# =========================================================
# 4. 기준 기간 설정
# =========================================================

# 2025년까지만 사용
df = df[df["연도"] <= 2025].copy()


# =========================================================
# 5. 연도별 관측일 수 계산
# =========================================================

year_days = (
    df.groupby("연도")
    .size()
    .reset_index(name="관측일수")
)


# =========================================================
# 6. 연도별 평균기온 계산
# =========================================================

annual = (
    df.groupby("연도")["평균기온"]
    .mean()
    .reset_index(name="연평균기온")
)

# 관측일 수와 합치기
annual = annual.merge(
    year_days,
    on="연도",
    how="left"
)


# =========================================================
# 7. 관측일이 300일 이상인 해만 사용
# =========================================================

annual = annual[annual["관측일수"] >= 300].copy()

# 연도순 정렬
annual = annual.sort_values("연도").reset_index(drop=True)


# =========================================================
# 8. 회귀에 사용할 데이터 확인
# =========================================================

if len(annual) < 2:
    st.error("회귀분석을 수행하기에 충분한 연도별 데이터가 없습니다.")
    st.stop()


# =========================================================
# 9. 독립변수 X
#    1908년부터 지난 연수
# =========================================================

annual["지난연수"] = annual["연도"] - 1908


# =========================================================
# 10. 선형회귀
# =========================================================

x = annual["지난연수"].to_numpy(dtype=float)
y = annual["연평균기온"].to_numpy(dtype=float)

# 1차 선형회귀
slope, intercept = np.polyfit(x, y, 1)

# 예측값
annual["회귀예측기온"] = intercept + slope * annual["지난연수"]


# =========================================================
# 11. 상관계수
# =========================================================

correlation = np.corrcoef(x, y)[0, 1]


# =========================================================
# 12. 회귀선 정보
# =========================================================

start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())
year_count = len(annual)


# =========================================================
# 13. 회귀식 표시
# =========================================================

st.subheader("📈 연평균 기온과 회귀 직선")

st.write(
    f"회귀식: **연평균기온 = {intercept:.4f} + "
    f"{slope:.4f} × (연도 - 1908)**"
)

st.write(
    f"상관계수: **{correlation:.4f}**"
)


# =========================================================
# 14. 회귀에 사용된 기간 정보
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "회귀에 사용된 연도 수",
        f"{year_count}개"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{start_year}년"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{end_year}년"
    )


# =========================================================
# 15. 산점도 + 회귀선
# =========================================================

fig = go.Figure()


# 실제 연평균 기온 산점도
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=annual["관측일수"],
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        )
    )
)


# 회귀선
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["회귀예측기온"],
        mode="lines",
        name="회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 예측기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


fig.update_layout(
    title="서울 연평균 기온과 선형회귀",
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified",
    height=600
)

# 가로축을 실제 연도로 표시
fig.update_xaxes(
    tickmode="auto",
    dtick=10
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 16. 연도 슬라이더
# =========================================================

st.subheader("🔮 연도별 예상 기온")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# =========================================================
# 17. 선택한 연도의 예상 기온 계산
# =========================================================

selected_x = selected_year - 1908

predicted_temperature = intercept + slope * selected_x


# =========================================================
# 18. 예상 기온 크게 표시
# =========================================================

st.markdown(
    f"""
    <div style="
        background-color: #f5f7fa;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 25px;
    ">
        <div style="
            font-size: 22px;
            color: #555;
            margin-bottom: 10px;
        ">
            {selected_year}년 예상 연평균 기온
        </div>

        <div style="
            font-size: 55px;
            font-weight: bold;
            color: #222;
        ">
            {predicted_temperature:.2f}℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 19. 선택한 연도의 회귀 예측 위치를 그래프에 표시
# =========================================================

prediction_fig = go.Figure()

prediction_fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=6),
        opacity=0.7
    )
)

# 1900~2100년 전체 회귀선
future_years = np.arange(1900, 2101)

future_x = future_years - 1908
future_predictions = intercept + slope * future_x

prediction_fig.add_trace(
    go.Scatter(
        x=future_years,
        y=future_predictions,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3)
    )
)

# 선택한 연도
prediction_fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temperature],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(
            size=16,
            symbol="star"
        ),
        hovertemplate=(
            f"{selected_year}년<br>"
            f"예상 기온: {predicted_temperature:.2f}℃"
            "<extra></extra>"
        )
    )
)

prediction_fig.update_layout(
    title=f"{selected_year}년의 회귀 예측 위치",
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    xaxis=dict(
        range=[1900, 2100]
    ),
    height=500
)

prediction_fig.update_xaxes(
    dtick=10
)

st.plotly_chart(
    prediction_fig,
    use_container_width=True
)


# =========================================================
# 20. 데이터 처리 기준 안내
# =========================================================

st.subheader("📌 데이터 처리 기준")

st.write(
    "• 2025년까지의 데이터만 사용했습니다."
)

st.write(
    "• 한 해의 관측일수가 300일 미만인 연도는 제외했습니다."
)

st.write(
    "• 남은 각 연도의 일평균기온을 평균하여 연평균기온을 계산했습니다."
)

st.write(
    "• 회귀의 독립변수는 '1908년부터 지난 연수(연도 - 1908)'입니다."
)

st.write(
    f"• 실제 회귀에 사용된 기간은 {start_year}년부터 {end_year}년까지이며, "
    f"총 {year_count}개 연도입니다."
)
