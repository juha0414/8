import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ============================================================
# 1. 기본 설정
# ============================================================

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연평균기온 데이터를 이용해 회귀 직선을 만들고, "
    "선택한 연도의 예상 평균기온을 확인합니다."
)


# ============================================================
# 2. 데이터 불러오기
# ============================================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

try:
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# ============================================================
# 3. 데이터 전처리
# ============================================================

# 날짜를 날짜 형식으로 변환
df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

# 평균기온을 숫자로 변환
df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

# 날짜 또는 평균기온이 없는 행 제거
df = df.dropna(subset=["날짜", "평균기온"]).copy()

# 연도 추출
df["연도"] = df["날짜"].dt.year


# ============================================================
# 4. 2025년까지의 데이터만 사용
# ============================================================

df = df[df["연도"] <= 2025].copy()


# ============================================================
# 5. 연도별 관측일 수와 연평균기온 계산
# ============================================================

yearly = (
    df.groupby("연도")
    .agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    )
    .reset_index()
)

# 관측일수가 300일 이상인 해만 사용
yearly = yearly[yearly["관측일수"] >= 300].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)


# ============================================================
# 6. 회귀 분석
#    독립변수 = 1908년부터 지난 연수
# ============================================================

# x = 1908년부터 지난 연수
yearly["경과연수"] = yearly["연도"] - 1908

x = yearly["경과연수"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)


if len(yearly) < 2:
    st.error("회귀선을 만들 수 있는 데이터가 충분하지 않습니다.")
    st.stop()


# 1차 선형 회귀
slope, intercept = np.polyfit(x, y, 1)

# 회귀선의 예측값
yearly["회귀예상기온"] = slope * yearly["경과연수"] + intercept

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]


# ============================================================
# 7. 회귀선 정보
# ============================================================

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
data_count = len(yearly)


# ============================================================
# 8. 화면에 회귀 분석 정보 표시
# ============================================================

st.subheader("📊 회귀 분석 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("회귀선에 사용한 해의 개수", f"{data_count}개")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")

with col4:
    st.metric("상관계수", f"{correlation:.3f}")


# ============================================================
# 9. 산점도 + 회귀 직선
# ============================================================

st.subheader("📈 연도별 연평균기온과 회귀 직선")

fig = go.Figure()


# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7
        ),
        customdata=yearly[["관측일수"]],
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)


# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["회귀예상기온"],
        mode="lines",
        name="회귀 직선",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 예상기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="x unified",
    height=600,
    legend_title="구분"
)

# 가로축에 연도를 그대로 표시
fig.update_xaxes(
    dtick=10,
    tickformat="d"
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 10. 회귀식 설명
# ============================================================

st.subheader("🧮 회귀식")

st.write(
    f"1908년부터 지난 연수를 x라고 하면, "
    f"회귀식은 다음과 같습니다."
)

st.latex(
    f"\\hat{{y}} = {intercept:.4f} + ({slope:.4f})x"
)

st.write(
    f"연도별 연평균기온과 경과연수의 상관계수는 "
    f"**{correlation:.3f}**입니다."
)


# ============================================================
# 11. 연도 슬라이더
# ============================================================

st.subheader("🔮 연도별 예상 기온")

selected_year = st.slider(
    "예상 기온을 확인할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# 선택한 연도의 경과연수
selected_elapsed_years = selected_year - 1908

# 회귀식으로 예상기온 계산
predicted_temperature = (
    intercept + slope * selected_elapsed_years
)


# ============================================================
# 12. 선택한 연도의 예상 기온 크게 표시
# ============================================================

st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 30px;
        border-radius: 15px;
        background-color: #f5f7fa;
        margin-top: 10px;
        margin-bottom: 20px;
    ">
        <div style="font-size: 24px; color: #555;">
            {selected_year}년 예상 연평균기온
        </div>
        <div style="
            font-size: 64px;
            font-weight: bold;
            margin-top: 10px;
        ">
            {predicted_temperature:.2f} °C
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 13. 선택한 연도의 정보
# ============================================================

if selected_year in yearly["연도"].values:
    actual_temperature = yearly.loc[
        yearly["연도"] == selected_year,
        "연평균기온"
    ].iloc[0]

    observation_days = yearly.loc[
        yearly["연도"] == selected_year,
        "관측일수"
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "선택한 연도",
            f"{selected_year}년"
        )

    with col2:
        st.metric(
            "실제 연평균기온",
            f"{actual_temperature:.2f} °C"
        )

    with col3:
        st.metric(
            "관측일수",
            f"{observation_days}일"
        )

else:
    st.info(
        "선택한 연도는 회귀선을 만드는 데 사용된 실제 관측 연도가 아닙니다. "
        "회귀 직선을 이용한 예상값만 표시합니다."
    )


# ============================================================
# 14. 데이터 기준 안내
# ============================================================

st.divider()

st.caption(
    "분석 기준: 2025년까지의 데이터 중 연간 관측일수가 300일 이상인 해만 사용"
)

st.caption(
    "회귀 분석의 독립변수는 '1908년부터 지난 연수'이며, "
    "종속변수는 해당 연도의 연평균기온입니다."
)

st.caption(
    "데이터 출처: 서울 기온 데이터(seoul.csv)"
)
