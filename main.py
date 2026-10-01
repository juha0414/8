
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
    "연도별 예상 기온을 확인합니다."
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
        st.error("데이터를 불러오는 중 오류가 발생했습니다.")
        st.exception(e)
        st.stop()


# ============================================================
# 3. 데이터 전처리
# ============================================================

df["날짜"] = pd.to_datetime(
    df["날짜"],
    errors="coerce"
)

df["평균기온"] = pd.to_numeric(
    df["평균기온"],
    errors="coerce"
)

# 날짜와 평균기온이 없는 자료 제거
df = df.dropna(
    subset=["날짜", "평균기온"]
).copy()

# 연도 추출
df["연도"] = df["날짜"].dt.year


# ============================================================
# 4. 2025년까지의 데이터만 사용
# ============================================================

df = df[
    df["연도"] <= 2025
].copy()


# ============================================================
# 5. 연도별 관측일수와 연평균기온 계산
# ============================================================

yearly = (
    df.groupby("연도")
    .agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    )
    .reset_index()
)

# 관측일수가 300일 미만인 해는 제외
yearly = yearly[
    yearly["관측일수"] >= 300
].copy()

# 연도순으로 정렬
yearly = yearly.sort_values(
    "연도"
).reset_index(drop=True)


if len(yearly) < 2:
    st.error(
        "회귀 분석에 사용할 수 있는 연도가 충분하지 않습니다."
    )
    st.stop()


# ============================================================
# 6. 전체 기간 회귀 분석
# ============================================================

# 독립변수:
# 1908년부터 몇 년이 지났는가
yearly["경과연수"] = (
    yearly["연도"] - 1908
)

x = yearly["경과연수"].to_numpy(
    dtype=float
)

y = yearly["연평균기온"].to_numpy(
    dtype=float
)

# 1차 선형 회귀
slope, intercept = np.polyfit(
    x,
    y,
    1
)

# 회귀선의 예상 기온
yearly["회귀예상기온"] = (
    intercept
    + slope * yearly["경과연수"]
)

# 상관계수
correlation = np.corrcoef(
    x,
    y
)[0, 1]

# 1년에 몇 도 변하는지 → 100년에 몇 도 변하는지
slope_100 = slope * 100


# ============================================================
# 7. 전체 기간 정보
# ============================================================

start_year = int(
    yearly["연도"].min()
)

end_year = int(
    yearly["연도"].max()
)

data_count = len(yearly)


# ============================================================
# 8. 최근 20년 회귀 분석
# ============================================================

# 마지막 분석 연도를 기준으로 최근 20년
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
    recent_slope_100 = (
        recent_slope * 100
    )

else:

    recent_slope = np.nan
    recent_intercept = np.nan
    recent_slope_100 = np.nan


# ============================================================
# 9. 100년에 몇 도 변하는가
# ============================================================

st.subheader(
    "🌡️ 100년에 기온이 얼마나 변하는가?"
)

st.metric(
    label="전체 기간 기준",
    value=f"{slope_100:+.2f} °C",
    delta="100년에 기온 변화량"
)


# ============================================================
# 10. 전체 기간 vs 최근 20년 비교
# ============================================================

st.subheader(
    "📊 전체 기간과 최근 20년의 기온 변화율 비교"
)

col1, col2 = st.columns(2)


# ----------------------------
# 전체 기간
# ----------------------------

with col1:

    st.markdown("### 전체 기간")

    st.metric(
        label="100년에 기온 변화량",
        value=f"{slope_100:+.2f} °C"
    )

    st.write(
        f"분석 기간: **{start_year}년 ~ {end_year}년**"
    )


# ----------------------------
# 최근 20년
# ----------------------------

with col2:

    st.markdown("### 최근 20년")

    if not np.isnan(recent_slope_100):

        st.metric(
            label="100년에 기온 변화량",
            value=f"{recent_slope_100:+.2f} °C"
        )

        st.write(
            f"분석 기간: "
            f"**{recent_start_year}년 ~ {end_year}년**"
        )

    else:

        st.warning(
            "최근 20년 회귀선을 계산할 데이터가 부족합니다."
        )


# ============================================================
# 11. 회귀 분석에 사용된 데이터 정보
# ============================================================

st.subheader(
    "📋 회귀선에 사용된 데이터"
)

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
# 12. 산점도 + 회귀 직선
# ============================================================

st.subheader(
    "📈 연도별 연평균기온과 회귀 직선"
)

fig = go.Figure()


# ------------------------------------------------------------
# 실제 연평균기온
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7
        ),
        customdata=yearly[
            ["관측일수"]
        ],
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)


# ------------------------------------------------------------
# 전체 기간 회귀 직선
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 최근 20년 회귀 직선
# ------------------------------------------------------------

if len(recent_20) >= 2:

    recent_line_x = np.array(
        [
            recent_start_year,
            end_year
        ],
        dtype=float
    )

    recent_line_elapsed = (
        recent_line_x - 1908
    )

    recent_line_y = (
        recent_intercept
        + recent_slope
        * recent_line_elapsed
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
                "최근 20년 회귀 예상기온: "
                "%{y:.2f} °C"
                "<extra></extra>"
            )
        )
    )


# ============================================================
# 13. 그래프 설정
# ============================================================

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
# 14. 회귀식
# ============================================================

st.subheader(
    "🧮 회귀식"
)

st.write(
    "전체 기간 회귀분석은 1908년부터 지난 연수를 "
    "독립변수로 사용합니다."
)

st.latex(
    f"\\hat{{y}} = "
    f"{intercept:.4f} "
    f"+ ({slope:.4f})x"
)

st.write(
    f"전체 기간 기울기: "
    f"**{slope_100:+.2f} °C / 100년**"
)


if len(recent_20) >= 2:

    st.write(
        f"최근 20년({recent_start_year}~{end_year}) "
        "회귀식:"
    )

    st.latex(
        f"\\hat{{y}} = "
        f"{recent_intercept:.4f} "
        f"+ ({recent_slope:.4f})x"
    )

    st.write(
        f"최근 20년 기울기: "
        f"**{recent_slope_100:+.2f} °C / 100년**"
    )


# ============================================================
# 15. 연도 슬라이더
# ============================================================

st.subheader(
    "🔮 연도별 예상 기온"
)

selected_year = st.slider(
    "예상 기온을 확인할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# ============================================================
# 16. 선택한 연도의 예상 기온 계산
# ============================================================

selected_elapsed_years = (
    selected_year - 1908
)

predicted_temperature = (
    intercept
    + slope * selected_elapsed_years
)


# ============================================================
# 17. 선택한 연도 예상 기온 크게 표시
# ============================================================

st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temperature:.2f} °C"
)


# ============================================================
# 18. 실제 관측자료가 있는 경우
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
# 19. 분석 기준
# ============================================================

st.divider()

st.caption(
    "분석 기준: 2025년까지의 데이터 중 "
    "연간 관측일수가 300일 이상인 해만 사용"
)

st.caption(
    "전체 기간 회귀분석: "
    "1908년부터 지난 연수를 독립변수로 사용"
)

st.caption(
    "최근 20년 회귀분석: "
    f"{recent_start_year}년부터 {end_year}년까지 사용"
)

st.caption(
    "기울기 단위: °C/년을 °C/100년으로 환산하여 표시"
)

st.caption(
    "데이터 출처: 서울 기온 데이터(seoul.csv)"
)
