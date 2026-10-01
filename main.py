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
    "서울의 연평균기온을 이용해 기온 변화 추세를 분석하고 "
    "미래 연도의 예상 기온을 확인합니다."
)


# ============================================================
# 2. 데이터 불러오기
# ============================================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

try:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
except Exception:
    try:
        df = pd.read_csv(DATA_URL, encoding="utf-8-sig")
    except Exception as e:
        st.error("데이터를 불러올 수 없습니다.")
        st.exception(e)
        st.stop()


# ============================================================
# 3. 데이터 전처리
# ============================================================

df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

df = df.dropna(subset=["날짜", "평균기온"]).copy()

df["연도"] = df["날짜"].dt.year


# ============================================================
# 4. 2025년까지의 데이터만 사용
# ============================================================

df = df[df["연도"] <= 2025].copy()


# ============================================================
# 5. 연도별 연평균기온 계산
# ============================================================

yearly = (
    df.groupby("연도")
    .agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    )
    .reset_index()
)

# 관측일수가 300일 미만인 해 제외
yearly = yearly[yearly["관측일수"] >= 300].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)


if len(yearly) < 2:
    st.error("회귀 분석에 사용할 데이터가 충분하지 않습니다.")
    st.stop()


# ============================================================
# 6. 전체 기간 회귀 분석
# ============================================================

# 독립변수 = 1908년부터 지난 연수
yearly["경과연수"] = yearly["연도"] - 1908

x = yearly["경과연수"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)

yearly["회귀예상기온"] = (
    intercept + slope * yearly["경과연수"]
)

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# 100년 동안의 기온 변화량
slope_100 = slope * 100


# ============================================================
# 7. 전체 기간 정보
# ============================================================

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
data_count = len(yearly)


# ============================================================
# 8. 최근 20년 회귀 분석
# ============================================================

# 회귀에 사용된 마지막 연도를 기준으로 최근 20년 선택
recent_start_year = end_year - 19

recent_20 = yearly[
    yearly["연도"] >= recent_start_year
].copy()


if len(recent_20) >= 2:

    recent_x = (
        recent_20["연도"] - 1908
    ).to_numpy(dtype=float)

    recent_y = (
        recent_20["연평균기온"]
        .to_numpy(dtype=float)
    )

    recent_slope, recent_intercept = np.polyfit(
        recent_x,
        recent_y,
        1
    )

    # 최근 20년 기울기를 100년 기준으로 환산
    recent_slope_100 = recent_slope * 100

else:
    recent_slope = np.nan
    recent_intercept = np.nan
    recent_slope_100 = np.nan


# ============================================================
# 9. 100년당 기온 변화량 크게 표시
# ============================================================

st.subheader("🌡️ 100년에 기온이 얼마나 변하는가?")

st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 35px 20px;
        border-radius: 18px;
        background-color: #f5f7fa;
        margin-top: 15px;
        margin-bottom: 25px;
    ">
        <div style="
            font-size: 25px;
            color: #555;
            margin-bottom: 10px;
        ">
            전체 기간 기준
        </div>

        <div style="
            font-size: 58px;
            font-weight: bold;
        ">
            {slope_100:+.2f} °C
        </div>

        <div style="
            font-size: 20px;
            color: #666;
            margin-top: 8px;
        ">
            100년에 기온 변화량
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 10. 전체 기간 vs 최근 20년 비교
# ============================================================

st.subheader("📊 전체 기간과 최근 20년의 기온 변화율 비교")

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        f"""
        <div style="
            border: 2px solid #ddd;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
        ">
            <div style="
                font-size: 22px;
                font-weight: bold;
                margin-bottom: 15px;
            ">
                전체 기간
            </div>

            <div style="
                font-size: 48px;
                font-weight: bold;
            ">
                {slope_100:+.2f} °C
            </div>

            <div style="
                font-size: 18px;
                color: #666;
                margin-top: 10px;
            ">
                100년에 변화
            </div>

            <div style="
                font-size: 15px;
                color: #777;
                margin-top: 8px;
            ">
                {start_year}년 ~ {end_year}년
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    if not np.isnan(recent_slope_100):

        st.markdown(
            f"""
            <div style="
                border: 2px solid #ddd;
                border-radius: 15px;
                padding: 25px;
                text-align: center;
            ">
                <div style="
                    font-size: 22px;
                    font-weight: bold;
                    margin-bottom: 15px;
                ">
                    최근 20년
                </div>

                <div style="
                    font-size: 48px;
                    font-weight: bold;
                ">
                    {recent_slope_100:+.2f} °C
                </div>

                <div style="
                    font-size: 18px;
                    color: #666;
                    margin-top: 10px;
                ">
                    100년에 변화
                </div>

                <div style="
                    font-size: 15px;
                    color: #777;
                    margin-top: 8px;
                ">
                    {recent_start_year}년 ~ {end_year}년
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.warning("최근 20년 회귀선을 계산할 데이터가 부족합니다.")


# ============================================================
# 11. 회귀 분석 기본 정보
# ============================================================

st.subheader("📋 회귀선에 사용된 데이터")

info1, info2, info3, info4 = st.columns(4)

with info1:
    st.metric(
        "사용한 해의 개수",
        f"{data_count}개"
    )

with info2:
    st.metric(
        "시작 연도",
        f"{start_year}년"
    )

with info3:
    st.metric(
        "끝 연도",
        f"{end_year}년"
    )

with info4:
    st.metric(
        "상관계수",
        f"{correlation:.3f}"
    )


# ============================================================
# 12. 산점도 + 전체 기간 회귀 직선
# ============================================================

st.subheader("📈 연도별 연평균기온과 회귀 직선")

fig = go.Figure()


# 실제 연평균기온
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


# 전체 기간 회귀 직선
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["회귀예상기온"],
        mode="lines",
        name="전체 기간 회귀 직선",
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


# 최근 20년 회귀 직선
if len(recent_20) >= 2:

    recent_line_x = np.array(
        [recent_start_year, end_year],
        dtype=float
    )

    recent_line_elapsed = recent_line_x - 1908

    recent_line_y = (
        recent_intercept
        + recent_slope * recent_line_elapsed
    )

    fig.add_trace(
        go.Scatter(
            x=recent_line_x,
            y=recent_line_y,
            mode="lines",
            name="최근 20년 회귀 직선",
            line=dict(
                width=3,
                dash="dash"
            ),
            hovertemplate=(
                "연도: %{x:.0f}년<br>"
                "최근 20년 회귀 예상기온: %{y:.2f} °C"
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

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 13. 회귀식
# ============================================================

st.subheader("🧮 회귀식")

st.write(
    "전체 기간 회귀식은 1908년부터 지난 연수를 독립변수로 사용합니다."
)

st.latex(
    f"\\hat{{y}} = {intercept:.4f} + ({slope:.4f})x"
)

if len(recent_20) >= 2:

    st.write(
        f"최근 20년({recent_start_year}~{end_year})의 "
        "회귀식은 다음과 같습니다."
    )

    st.latex(
        f"\\hat{{y}} = {recent_intercept:.4f} "
        f"+ ({recent_slope:.4f})x"
    )


# ============================================================
# 14. 연도 슬라이더
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


# 전체 기간 회귀식으로 예상
predicted_temperature = (
    intercept
    + slope * selected_elapsed_years
)


# ============================================================
# 15. 예상 기온 크게 표시
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

        <div style="
            font-size: 24px;
            color: #555;
        ">
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
# 16. 실제 데이터가 있는 경우 비교
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
        "선택한 연도는 실제 관측자료가 없거나 "
        "회귀 분석에 사용되지 않은 연도입니다. "
        "회귀선을 이용한 예상값만 표시합니다."
    )


# ============================================================
# 17. 분석 기준
# ============================================================

st.divider()

st.caption(
    "분석 기준: 2025년까지의 데이터 중 연간 관측일수가 300일 이상인 해만 사용"
)

st.caption(
    "전체 기간 회귀분석: 1908년부터 지난 연수를 독립변수로 사용"
)

st.caption(
    "최근 20년 회귀분석: 회귀분석에 사용된 마지막 연도를 기준으로 "
    "최근 20년의 자료를 사용"
)

st.caption(
    "기울기의 단위: °C/년을 °C/100년으로 환산하여 표시"
)

st.caption(
    "데이터 출처: 서울 기온 데이터(seoul.csv)"
)
