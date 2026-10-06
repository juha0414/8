import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. 페이지 설정
# =========================================================

st.set_page_config(
    page_title="회귀모델 성능평가",
    page_icon="📈",
    layout="wide"
)

st.title("📈 기온 예측 회귀모델 성능평가")

st.write(
    "과거 연도의 연평균 기온으로 선형회귀 모델을 학습하고, "
    "최근 20년(2006~2025)을 공통 테스트 데이터로 사용하여 "
    "모델의 예측 성능을 비교합니다."
)


# =========================================================
# 2. 데이터 불러오기
# =========================================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

df = pd.read_csv(DATA_URL, encoding="utf-8")


# =========================================================
# 3. 데이터 전처리
# =========================================================

df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

df = df.dropna(subset=["날짜", "평균기온"])

# 2025년까지 사용
df = df[df["날짜"].dt.year <= 2025]

# 연도 추출
df["연도"] = df["날짜"].dt.year

# 연도별 관측일 수
year_count = df.groupby("연도").size()

# 300일 이상 관측된 연도만 사용
valid_years = year_count[year_count >= 300].index

df = df[df["연도"].isin(valid_years)]


# =========================================================
# 4. 연평균 기온 계산
# =========================================================

yearly = (
    df.groupby("연도")["평균기온"]
    .mean()
    .reset_index()
)

yearly.columns = ["연도", "연평균기온"]

yearly = yearly.sort_values("연도").reset_index(drop=True)

# 기준 연도부터 몇 년이 지났는지
yearly["경과연수"] = yearly["연도"] - 1908


# =========================================================
# 5. 사용할 데이터 구간
# =========================================================

# 1956~2005 : 최근 50년 학습
train_50 = yearly[
    (yearly["연도"] >= 1956) &
    (yearly["연도"] <= 2005)
].copy()

# 1906~2005 : 최근 100년 학습
train_100 = yearly[
    (yearly["연도"] >= 1906) &
    (yearly["연도"] <= 2005)
].copy()

# 2006~2025 : 공통 테스트
test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()


# =========================================================
# 6. 데이터 확인
# =========================================================

st.subheader("📊 학습 데이터와 테스트 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "50년 학습 데이터",
        f"{len(train_50)}개 연도"
    )
    if len(train_50) > 0:
        st.caption(
            f"{train_50['연도'].min()} ~ {train_50['연도'].max()}"
        )

with col2:
    st.metric(
        "100년 학습 데이터",
        f"{len(train_100)}개 연도"
    )
    if len(train_100) > 0:
        st.caption(
            f"{train_100['연도'].min()} ~ {train_100['연도'].max()}"
        )

with col3:
    st.metric(
        "공통 테스트 데이터",
        f"{len(test)}개 연도"
    )
    if len(test) > 0:
        st.caption(
            f"{test['연도'].min()} ~ {test['연도'].max()}"
        )


# =========================================================
# 7. 회귀모델 만들기
# =========================================================

X_50 = train_50[["경과연수"]]
y_50 = train_50["연평균기온"]

X_100 = train_100[["경과연수"]]
y_100 = train_100["연평균기온"]

X_test = test[["경과연수"]]
y_test = test["연평균기온"]


model_50 = LinearRegression()
model_100 = LinearRegression()

model_50.fit(X_50, y_50)
model_100.fit(X_100, y_100)


# =========================================================
# 8. 테스트 데이터 예측
# =========================================================

pred_50 = model_50.predict(X_test)
pred_100 = model_100.predict(X_test)


# =========================================================
# 9. 성능 평가
# =========================================================

mae_50 = mean_absolute_error(y_test, pred_50)
mse_50 = mean_squared_error(y_test, pred_50)
r2_50 = r2_score(y_test, pred_50)

mae_100 = mean_absolute_error(y_test, pred_100)
mse_100 = mean_squared_error(y_test, pred_100)
r2_100 = r2_score(y_test, pred_100)


# =========================================================
# 10. 기울기 계산
# =========================================================

slope_50 = model_50.coef_[0]
slope_100 = model_100.coef_[0]

slope_50_100 = slope_50 * 100
slope_100_100 = slope_100 * 100


# =========================================================
# 11. 회귀선 기울기 비교
# =========================================================

st.subheader("📈 회귀선의 기울기 비교")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 최근 50년 학습")
    st.metric(
        "100년에 기온이 얼마나 변하는가?",
        f"{slope_50_100:.2f} °C"
    )
    st.write(
        f"1년당 기울기: {slope_50:.4f} °C"
    )

with col2:
    st.markdown("### 최근 100년 학습")
    st.metric(
        "100년에 기온이 얼마나 변하는가?",
        f"{slope_100_100:.2f} °C"
    )
    st.write(
        f"1년당 기울기: {slope_100:.4f} °C"
    )


# =========================================================
# 12. 예측 성능 비교
# =========================================================

st.subheader("🎯 최근 20년 테스트 데이터 예측 성능")

comparison = pd.DataFrame({
    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
    ],
    "MAE": [
        mae_50,
        mae_100
    ],
    "MSE": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

comparison_display = comparison.copy()

comparison_display["MAE"] = comparison_display["MAE"].map(
    lambda x: f"{x:.3f}"
)

comparison_display["MSE"] = comparison_display["MSE"].map(
    lambda x: f"{x:.3f}"
)

comparison_display["R²"] = comparison_display["R²"].map(
    lambda x: f"{x:.3f}"
)

st.table(comparison_display)


st.write("")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("50년 MAE", f"{mae_50:.3f}")

with col2:
    st.metric("100년 MAE", f"{mae_100:.3f}")

with col3:
    st.metric("50년 MSE", f"{mse_50:.3f}")

with col4:
    st.metric("100년 MSE", f"{mse_100:.3f}")

with col5:
    st.metric("50년 R²", f"{r2_50:.3f}")

with col6:
    st.metric("100년 R²", f"{r2_100:.3f}")


# =========================================================
# 13. 실제값과 예측값 그래프
# =========================================================

st.subheader("📊 최근 20년 실제 기온과 모델 예측 비교")

fig = go.Figure()

# 실제값
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 연평균 기온"
    )
)

# 50년 모델
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines+markers",
        name="50년 학습 모델 예측"
    )
)

# 100년 모델
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_100,
        mode="lines+markers",
        name="100년 학습 모델 예측"
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (°C)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 14. 연도별 예측 결과
# =========================================================

st.subheader("🔎 최근 20년 연도별 예측 결과")

result = test[["연도", "연평균기온"]].copy()

result["50년 모델 예측"] = pred_50
result["100년 모델 예측"] = pred_100

result["50년 모델 오차"] = (
    result["연평균기온"] - result["50년 모델 예측"]
)

result["100년 모델 오차"] = (
    result["연평균기온"] - result["100년 모델 예측"]
)

result_display = result.copy()

for column in [
    "연평균기온",
    "50년 모델 예측",
    "100년 모델 예측",
    "50년 모델 오차",
    "100년 모델 오차"
]:
    result_display[column] = result_display[column].round(3)

st.dataframe(
    result_display,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 15. 어느 모델이 더 좋은지 판단
# =========================================================

st.subheader("🏆 모델 성능 비교 결과")

if mae_50 < mae_100:
    mae_result = "50년 학습 모델"
else:
    mae_result = "100년 학습 모델"

if mse_50 < mse_100:
    mse_result = "50년 학습 모델"
else:
    mse_result = "100년 학습 모델"

if r2_50 > r2_100:
    r2_result = "50년 학습 모델"
else:
    r2_result = "100년 학습 모델"

st.write(
    f"- **MAE가 더 낮은 모델:** {mae_result}"
)

st.write(
    f"- **MSE가 더 낮은 모델:** {mse_result}"
)

st.write(
    f"- **R²가 더 높은 모델:** {r2_result}"
)

st.info(
    "MAE와 MSE는 낮을수록 예측 오차가 작고, "
    "R²는 높을수록 실제 기온의 변화를 더 잘 설명하는 모델입니다."
)
