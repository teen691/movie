```python
# main.py — 영화 유형 나누기
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")
st.title("🎬 영화 유형 나누기")
st.caption("정답 열 없이, 내가 고른 속성이 비슷한 영화끼리 묶습니다.")

MOVIES = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

기호 = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]
속성 = ["스크린 수(로그)", "누적 관객(로그)", "10위권 일수", "롱런 지수"]
원래단위 = ["스크린 수", "누적 관객", "10위권 일수", "롱런 지수"]


@st.cache_data
def load_data():
    df = pd.read_csv(MOVIES, dtype={"movieCd": str})
    df = df.sort_values("movieCd").reset_index(drop=True)
    전체편수 = len(df)

    쓸열 = ["first_scrn", "total_audi", "days_in_top10", "first_week_audi"]
    df = df.dropna(subset=쓸열 + ["movieNm"])
    df = df[
        (df["first_scrn"] > 0)
        & (df["total_audi"] > 0)
        & (df["first_week_audi"] > 0)
    ].reset_index(drop=True)

    return df, 전체편수


df, 전체편수 = load_data()

# 분석에 사용할 속성 만들기
df["스크린 수"] = df["first_scrn"]
df["누적 관객"] = df["total_audi"]
df["10위권 일수"] = df["days_in_top10"]
df["롱런 지수"] = (
    df["total_audi"] / df["first_week_audi"]
).clip(upper=20)

df["스크린 수(로그)"] = np.log10(df["first_scrn"])
df["누적 관객(로그)"] = np.log10(df["total_audi"])

st.info(
    f"전체 {전체편수:,}편 가운데 네 속성을 모두 만들 수 있는 "
    f"{len(df):,}편으로 묶었습니다."
)

# 속성과 묶음 수 선택
왼쪽, 오른쪽 = st.columns(2)

with 왼쪽:
    고른속성 = st.multiselect(
        "묶는 데 사용할 속성 (둘 이상)",
        속성,
        default=속성
    )

with 오른쪽:
    k = st.selectbox(
        "묶음 수",
        options=list(range(2, 8)),
        index=1
    )

if len(고른속성) < 2:
    st.warning("속성을 둘 이상 골라 주세요.")
    st.stop()

# 선택한 속성을 표준화
Xs = StandardScaler().fit_transform(df[고른속성])

# 선택한 묶음 수로 군집화
모델 = KMeans(n_clusters=k, n_init=10, random_state=42)
번호 = 모델.fit_predict(Xs)

# 누적 관객 평균이 많은 묶음부터 기호 지정
순서 = (
    pd.Series(df["누적 관객"].to_numpy())
    .groupby(번호)
    .mean()
    .sort_values(ascending=False)
    .index
)

자리 = {g: i for i, g in enumerate(순서)}
df["묶음"] = [기호[자리[g]] for g in 번호]
현재기호 = 기호[:k]

# 2차원 산점도
st.subheader("묶음 지도 · 2차원")

c1, c2 = st.columns(2)

x축 = c1.selectbox("가로축", 고른속성, index=0)
y축 = c2.selectbox(
    "세로축",
    고른속성,
    index=min(1, len(고른속성) - 1)
)

fig = px.scatter(
    df,
    x=x축,
    y=y축,
    color="묶음",
    hover_name="movieNm",
    category_orders={"묶음": 현재기호}
)

fig.update_traces(marker=dict(size=9, opacity=0.8))
fig.update_layout(height=460)
st.plotly_chart(fig, width="stretch")

st.caption(
    "점 하나가 영화 한 편입니다. 축을 바꿔 보면 "
    "묶음이 나뉘는 축과 섞이는 축이 보입니다."
)

# 3차원 산점도
st.subheader("묶음 지도 · 3차원")

if len(고른속성) < 3:
    st.info("속성을 셋 이상 고르면 3차원 그림이 표시됩니다.")
else:
    d1, d2, d3 = st.columns(3)

    x3 = d1.selectbox("x축", 고른속성, index=0, key="x3")
    y3 = d2.selectbox("y축", 고른속성, index=1, key="y3")
    z3 = d3.selectbox("z축", 고른속성, index=2, key="z3")

    fig3 = px.scatter_3d(
        df,
        x=x3,
        y=y3,
        z=z3,
        color="묶음",
        hover_name="movieNm",
        category_orders={"묶음": 현재기호}
    )

    fig3.update_traces(marker=dict(size=2, opacity=0.7))
    fig3.update_layout(
        height=560,
        legend=dict(orientation="h")
    )

    st.plotly_chart(fig3, width="stretch")
    st.caption("마우스로 끌면 돌아갑니다. 점에 마우스를 올리면 제목이 보입니다.")

# 묶음별 요약 표
st.subheader("묶음별 편수와 네 속성의 평균")

요약 = df.groupby("묶음")[원래단위].mean().round(2)
요약.insert(0, "편수", df.groupby("묶음").size())
요약 = 요약.reindex(현재기호)

st.dataframe(
    요약,
    width="stretch",
    column_config={
        "편수": st.column_config.NumberColumn("편수", format="%,d편"),
        "스크린 수": st.column_config.NumberColumn("스크린 수", format="%.1f개"),
        "누적 관객": st.column_config.NumberColumn("누적 관객", format="%,.0f명"),
        "10위권 일수": st.column_config.NumberColumn("10위권 일수", format="%.1f일"),
        "롱런 지수": st.column_config.NumberColumn("롱런 지수", format="%.2f배"),
    }
)

st.caption(
    "평균은 로그를 취하기 전의 원래 단위입니다. "
    "롱런 지수는 누적 관객을 첫 주 관객으로 나눈 값이고 20에서 잘랐습니다."
)

# 묶음마다 관객이 많은 다섯 편
st.subheader("묶음마다 관객이 많은 다섯 편")

열 = st.columns(k)

for i, 이름 in enumerate(현재기호):
    한묶음 = (
        df[df["묶음"] == 이름]
        .sort_values("누적 관객", ascending=False)
        .head(5)
    )

    전체묶음편수 = int((df["묶음"] == 이름).sum())

    열[i].markdown(f"**{이름} 묶음 ({전체묶음편수:,}편)**")

    열[i].dataframe(
        pd.DataFrame({
            "영화": 한묶음["movieNm"].to_numpy(),
            "누적 관객": 한묶음["누적 관객"].to_numpy()
        }),
        width="stretch",
        hide_index=True,
        column_config={
            "누적 관객": st.column_config.NumberColumn(
                "누적 관객",
                format="%,d명"
            )
        }
    )

# 묶음 수 1~7의 군집 내 거리 제곱합 계산
st.subheader("묶음 수에 따른 군집 내 거리 제곱합")

st.caption(
    "값이 작을수록 각 점이 자기 묶음의 중심에 가깝습니다. "
    "묶음 수를 늘릴 때 값이 얼마나 줄어드는지 살펴보세요."
)

wcss = []

for 군집수 in range(1, 8):
    평가모델 = KMeans(
        n_clusters=군집수,
        n_init=10,
        random_state=42
    )
    평가모델.fit(Xs)
    wcss.append(float(평가모델.inertia_))

wcss표 = pd.DataFrame({
    "묶음 수": list(range(1, 8)),
    "거리 제곱합": wcss
})

wcss표["바로 앞 값보다 감소한 양"] = [
    np.nan
] + [
    wcss[i - 1] - wcss[i]
    for i in range(1, len(wcss))
]

# 거리 제곱합 꺾은선 그래프
선그래프 = go.Figure()

선그래프.add_trace(
    go.Scatter(
        x=wcss표["묶음 수"],
        y=wcss표["거리 제곱합"],
        mode="lines+markers",
        name="거리 제곱합",
        hovertemplate=(
            "묶음 수: %{x}<br>"
            "거리 제곱합: %{y:,.2f}<extra></extra>"
        )
    )
)

# 현재 선택한 묶음 수에 세로선 표시
선그래프.add_vline(
    x=k,
    line_dash="dash",
    line_width=2,
    annotation_text=f"현재 선택: {k}개",
    annotation_position="top"
)

선그래프.update_layout(
    xaxis_title="묶음 수",
    yaxis_title="군집 내 거리 제곱합(WCSS)",
    xaxis=dict(
        tickmode="linear",
        dtick=1,
        range=[0.8, 7.2]
    ),
    height=430,
    margin=dict(t=65)
)

st.plotly_chart(선그래프, width="stretch")

# 거리 제곱합 비교 표
st.dataframe(
    wcss표,
    width="stretch",
    hide_index=True,
    column_config={
        "묶음 수": st.column_config.NumberColumn(
            "묶음 수",
            format="%d개"
        ),
        "거리 제곱합": st.column_config.NumberColumn(
            "거리 제곱합",
            format="%,.2f"
        ),
        "바로 앞 값보다 감소한 양": st.column_config.NumberColumn(
            "바로 앞 값보다 감소한 양",
            format="%,.2f"
        ),
    }
)

# 현재 묶음 수의 실루엣 점수
st.subheader("현재 묶음 수의 실루엣 점수")

실루엣 = None

평가모델 = KMeans(
    n_clusters=k,
    n_init=10,
    random_state=42
)
평가번호 = 평가모델.fit_predict(Xs)

고유묶음수 = len(np.unique(평가번호))

if 2 <= 고유묶음수 < len(df):
    실루엣 = float(silhouette_score(Xs, 평가번호))

if 실루엣 is None:
    st.info("현재 군집 결과에서는 실루엣 점수를 계산할 수 없습니다.")
else:
    st.markdown(
        f"**묶음 수 {k}개 · 실루엣 점수: {실루엣:.3f}**"
    )

    st.caption(
        "실루엣 점수는 -1에서 1 사이입니다. "
        "1에 가까울수록 묶음이 서로 잘 구분됩니다. "
        "0에 가까우면 묶음이 겹칠 수 있고, 음수이면 "
        "일부 영화가 다른 묶음에 더 가까울 수 있습니다."
    )
```
