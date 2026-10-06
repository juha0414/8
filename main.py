
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
st.write(
    "서울의 연평균 기온 데이터를 이용해 연도별 기온 변화를 분석하고 "
    "선형회귀를 이용해 기온을 예측합니다."
)


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

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 필요한 값이 없는 행 제거
    df = df.dropna(
        subset=["날짜", "평균기온"]
    ).copy()

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
# 4. 2025년까지만 사용
# =========================================================

df = df[
    df["연도"] <= 2025
].copy()


# =========================================================
# 5. 연도별 관측일 수
# =========================================================

year_days = (
    df.groupby("연도")
    .size()
    .reset_index(name="관측일수")
)


# =========================================================
# 6. 연도별 평균기온
# =========================================================

annual = (
    df.groupby("연도")["평균기온"]
    .mean()
    .reset_index(name="연평균기온")
)


# 관측일 수 합치기
annual = annual.merge(
    year_days,
    on="연도",
    how="left"
)


# =========================================================
# 7. 관측일수가 300일 미만인 해 제외
# =========================================================

annual = annual[
    annual["관측일수"] >= 300
].copy()


annual = annual.sort_values(
    "연도"
).reset_index(drop=True)


# =========================================================
# 8. 데이터 확인
# =========================================================

if len(annual) < 2:
    st.error(
        "회귀분석을 수행하기에 충분한 데이터가 없습니다."
    )
    st.stop()


# =========================================================
# 9. 전체 기간 회귀
#
# 독립변수 = 1908년부터 지난 연수
# =========================================================

annual["지난연수"] = (
    annual["연도"] - 1908
)


x_all = annual["지난연수"].to_numpy(
    dtype=float
)

y_all = annual["연평균기온"].to_numpy(
    dtype=float
)


# 전체 기간 회귀
slope_all, intercept_all = np.polyfit(
    x_all,
    y_all,
    1
)


# 전체 기간 회귀 예측값
annual["전체기간_회귀예측"] = (
    intercept_all
    + slope_all * annual["지난연수"]
)


# =========================================================
# 10. 전체 기간 상관계수
# =========================================================

correlation = np.corrcoef(
    x_all,
    y_all
)[0, 1]


# =========================================================
# 11. 최근 20년 데이터
# =========================================================

latest_year = int(
    annual["연도"].max()
)

recent_start_year = (
    latest_year - 19
)

recent20 = annual[
    annual["연도"] >= recent_start_year
].copy()


# 최근 20년 데이터가 충분한지 확인
if len(recent20) >= 2:

    recent20["지난연수"] = (
        recent20["연도"] - 1908
    )

    x_recent = recent20[
        "지난연수"
    ].to_numpy(dtype=float)

    y_recent = recent20[
        "연평균기온"
    ].to_numpy(dtype=float)

    # 최근 20년 회귀
    slope_recent, intercept_recent = np.polyfit(
        x_recent,
        y_recent,
        1
    )

else:

    slope_recent = np.nan
    intercept_recent = np.nan


# =========================================================
# 12. 100년에 몇 도 오르는가 계산
# =========================================================

# slope는 ℃/년
# 따라서 100을 곱하면 ℃/100년

rise_100_all = slope_all * 100

if not np.isnan(slope_recent):
    rise_100_recent = slope_recent * 100
else:
    rise_100_recent = np.nan


# =========================================================
# 13. 기본 회귀 정보
# =========================================================

start_year = int(
    annual["연도"].min()
)

end_year = int(
    annual["연도"].max()
)

year_count = len(annual)


# =========================================================
# 14. 회귀 기울기 크게 표시
# =========================================================

st.subheader(
    "🌡️ 100년에 몇 도 오르는가?"
)

st.write(
    "연도별 평균기온에 선형회귀를 적용한 결과입니다."
)


col1, col2 = st.columns(2)


# ---------------------------------------------------------
# 전체 기간
# ---------------------------------------------------------

with col1:

    st.markdown(
        f"""
        <div style="
            border: 2px solid #dddddd;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            background-color: #f8f9fa;
        ">

            <div style="
                font-size: 20px;
                color: #555555;
                margin-bottom: 10px;
            ">
                전체 기간
            </div>

            <div style="
                font-size: 42px;
                font-weight: bold;
                margin-bottom: 10px;
            ">
                {rise_100_all:.2f}℃
            </div>

            <div style="
                font-size: 16px;
                color: #666666;
            ">
                100년에 기온이 오르는 정도
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# 최근 20년
# ---------------------------------------------------------

with col2:

    if not np.isnan(rise_100_recent):

        st.markdown(
            f"""
            <div style="
                border: 2px solid #dddddd;
                border-radius: 15px;
                padding: 25px;
                text-align: center;
                background-color: #f8f9fa;
            ">

                <div style="
                    font-size: 20px;
                    color: #555555;
                    margin-bottom: 10px;
                ">
                    최근 20년
                </div>

                <div style="
                    font-size: 42px;
                    font-weight: bold;
                    margin-bottom: 10px;
                ">
                    {rise_100_recent:.2f}℃
                </div>

                <div style="
                    font-size: 16px;
                    color: #666666;
                ">
                    100년에 기온이 오르는 정도
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.warning(
            "최근 20년 회귀를 계산할 데이터가 부족합니다."
        )


# =========================================================
# 15. 전체 기간과 최근 20년 비교
# =========================================================

st.subheader(
    "📊 전체 기간 vs 최근 20년"
)

if not np.isnan(rise_100_recent):

    difference = (
        rise_100_recent
        - rise_100_all
    )

    comparison_col1, comparison_col2, comparison_col3 = st.columns(3)

    with comparison_col1:

        st.metric(
            "전체 기간",
            f"{rise_100_all:.2f} ℃ / 100년"
        )

    with comparison_col2:

        st.metric(
            "최근 20년",
            f"{rise_100_recent:.2f} ℃ / 100년"
        )

    with comparison_col3:

        st.metric(
            "최근 20년 - 전체 기간",
            f"{difference:+.2f} ℃ / 100년"
        )


# =========================================================
# 16. 회귀 기간 정보
# =========================================================

st.subheader(
    "📌 전체 기간 회귀에 사용된 데이터"
)

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:

    st.metric(
        "사용한 연도 수",
        f"{year_count}개"
    )

with info_col2:

    st.metric(
        "시작 연도",
        f"{start_year}년"
    )

with info_col3:

    st.metric(
        "끝 연도",
        f"{end_year}년"
    )


st.write(
    f"최근 20년 회귀는 "
    f"{recent_start_year}년부터 {latest_year}년까지의 "
    f"데이터를 사용했습니다."
)


# =========================================================
# 17. 회귀식과 상관계수
# =========================================================

st.subheader(
    "📈 회귀분석 결과"
)

st.write(
    f"전체 기간 회귀식: "
    f"연평균기온 = {intercept_all:.4f} + "
    f"{slope_all:.4f} × (연도 - 1908)"
)

st.write(
    f"전체 기간 상관계수: **{correlation:.4f}**"
)


if not np.isnan(slope_recent):

    st.write(
        f"최근 20년 회귀식: "
        f"연평균기온 = {intercept_recent:.4f} + "
        f"{slope_recent:.4f} × (연도 - 1908)"
    )


# =========================================================
# 18. 산점도 + 전체 기간 회귀선
# =========================================================

st.subheader(
    "📈 서울 연평균 기온과 회귀 직선"
)

fig = go.Figure()


# 실제 연평균 기온
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


# 전체 기간 회귀선
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["전체기간_회귀예측"],
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 예측기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 최근 20년 회귀선
if not np.isnan(slope_recent):

    recent_prediction = (
        intercept_recent
        + slope_recent
        * recent20["지난연수"]
    )

    fig.add_trace(
        go.Scatter(
            x=recent20["연도"],
            y=recent_prediction,
            mode="lines",
            name="최근 20년 회귀선",
            line=dict(
                width=3,
                dash="dash"
            ),
            hovertemplate=(
                "연도: %{x}년<br>"
                "최근 20년 회귀 예측: %{y:.2f}℃"
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


# 실제 연도 표시
fig.update_xaxes(
    dtick=10
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 19. 연도 슬라이더
# =========================================================

st.subheader(
    "🔮 연도별 예상 기온"
)

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# =========================================================
# 20. 선택한 연도의 예상 기온
# =========================================================

selected_x = (
    selected_year - 1908
)

predicted_temperature = (
    intercept_all
    + slope_all * selected_x
)


# =========================================================
# 21. 예상 기온 크게 표시
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
            color: #555555;
            margin-bottom: 10px;
        ">
            {selected_year}년 예상 연평균 기온
        </div>

        <div style="
            font-size: 55px;
            font-weight: bold;
            color: #222222;
        ">
            {predicted_temperature:.2f}℃
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 22. 1900~2100년 회귀선 + 선택 연도
# =========================================================

prediction_fig = go.Figure()


# 실제 데이터
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


# 1900~2100년 회귀선
future_years = np.arange(
    1900,
    2101
)

future_x = (
    future_years - 1908
)

future_predictions = (
    intercept_all
    + slope_all * future_x
)


prediction_fig.add_trace(
    go.Scatter(
        x=future_years,
        y=future_predictions,
        mode="lines",
        name="전체 기간 회귀선",
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
# 23. 데이터 처리 기준
# =========================================================

st.subheader(
    "📌 데이터 처리 기준"
)

st.write(
    "• 2025년까지의 데이터만 사용했습니다."
)

st.write(
    "• 한 해의 관측일수가 300일 미만인 연도는 제외했습니다."
)

st.write(
    "• 각 연도의 일평균기온을 평균하여 연평균기온을 계산했습니다."
)

st.write(
    "• 회귀의 독립변수는 1908년부터 지난 연수(연도 - 1908)입니다."
)

st.write(
    f"• 전체 기간 회귀: {start_year}년 ~ {end_year}년, "
    f"총 {year_count}개 연도"
)

st.write(
    f"• 최근 20년 회귀: {recent_start_year}년 ~ {latest_year}년"
)
