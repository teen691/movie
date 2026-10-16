# main.py — 영화 유형 나누기: 정답 없이 비슷한 영화끼리 묶는다
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")
st.title("🎬 영화 유형 나누기")
st.caption("정답 열 없이, 내가 고른 속성이 비슷한 영화끼리 묶습니다.")

MOVIES = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
기호 = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]          # 묶음 번호 대신 사용하는 표시
속성 = ["스크린 수(로그)", "누적 관객(로그)", "10위권 일수", "롱런 지수"]
원래단위 = ["스크린 수", "누적 관객", "10위권 일수", "롱런 지수"]


@st.cache_data
def load_data():
    df = pd.read_csv(MOVIES, dtype={"movieCd": str})
    df = df.sort_values("movieCd").reset_index(drop=True)
    전체편수 = len(df)
    # 네 속성을 만들 수 없는 영화는 뺀다: 값이 없거나 첫 주 관객이 0인 행
    쓸열 = ["first_scrn", "total_audi", "days_in_top10", "first_week_audi"]
    df = df.dropna(subset=쓸열 + ["movieNm"])
    df = df[(df["first_scrn"] > 0) & (df["total_audi"] > 0) & (df["first_week_audi"] > 0)]
    df = df.reset_index(drop=True)
    return df, 전체편수


df, 전체편수 = load_data()

# 네 가지 속성을 만든다. 숫자 차이가 너무 큰 둘은 상용로그를 취한다
df["스크린 수"] = df["first_scrn"]
df["누적 관객"] = df["total_audi"]
df["10위권 일수"] = df["days_in_top10"]
df["롱런 지수"] = (df["total_audi"] / df["first_week_audi"]).clip(upper=20)   # 20을 넘으면 20으로
df["스크린 수(로그)"] = np.log10(df["first_scrn"])
df["누적 관객(로그)"] = np.log10(df["total_audi"])

st.info(f"전체 {전체편수:,}편 가운데 네 속성을 모두 만들 수 있는 {len(df):,}편으로 묶었습니다.")

고른속성 = st.multiselect("묶는 데 사용할 속성 (둘 이상)", 속성, default=속성)
if len(고른속성) < 2:
    st.warning("속성을 둘 이상 골라 주세요.")
    st.stop()

# 고른 속성을 표준화한 뒤 k-평균으로 나눈다. 난수를 고정해 다시 실행해도 같은 결과가 나오게 한다
k = 3
Xs = StandardScaler().fit_transform(df[고른속성])
번호 = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xs)

# 누적 관객 평균이 큰 묶음부터 ㉮·㉯·㉰ 순으로 표시한다
순서 = pd.Series(df["누적 관객"].to_numpy()).groupby(번호).mean().sort_values(ascending=False).index
자리 = {g: i for i, g in enumerate(순서)}
df["묶음"] = [기호[자리[g]] for g in 번호]

st.subheader("묶음 지도 · 2차원")
c1, c2 = st.columns(2)
x축 = c1.selectbox("가로축", 고른속성, index=0)
y축 = c2.selectbox("세로축", 고른속성, index=min(1, len(고른속성) - 1))
fig = px.scatter(df, x=x축, y=y축, color="묶음", hover_name="movieNm",
                 category_orders={"묶음": 기호[:k]})
fig.update_traces(marker=dict(size=9, opacity=0.8))
fig.update_layout(height=460)
st.plotly_chart(fig, width="stretch")
st.caption("점 하나가 영화 한 편입니다. 축을 바꿔 보면 묶음이 나뉘는 축과 섞이는 축이 보입니다.")

st.subheader("묶음 지도 · 3차원")
if len(고른속성) < 3:
    st.info("속성을 셋 이상 고르면 3차원 그림이 표시됩니다.")
else:
    d1, d2, d3 = st.columns(3)
    x3 = d1.selectbox("x축", 고른속성, index=0, key="x3")
    y3 = d2.selectbox("y축", 고른속성, index=1, key="y3")
    z3 = d3.selectbox("z축", 고른속성, index=2, key="z3")
    fig3 = px.scatter_3d(df, x=x3, y=y3, z=z3, color="묶음", hover_name="movieNm",
                         category_orders={"묶음": 기호[:k]})
    fig3.update_traces(marker=dict(size=2, opacity=0.7))
    fig3.update_layout(height=560, legend=dict(orientation="h"))
    st.plotly_chart(fig3, width="stretch")
    st.caption("마우스로 끌면 돌아갑니다. 점에 마우스를 올리면 제목이 보입니다.")

st.subheader("묶음별 편수와 네 속성의 평균")
요약 = df.groupby("묶음")[원래단위].mean().round(2)
요약.insert(0, "편수", df.groupby("묶음").size())
요약 = 요약.reindex(기호[:k])
st.dataframe(요약, width="stretch", column_config={
    "편수": st.column_config.NumberColumn("편수", format="%,d편"),
    "스크린 수": st.column_config.NumberColumn("스크린 수", format="%.1f개"),
    "누적 관객": st.column_config.NumberColumn("누적 관객", format="%,.0f명"),
    "10위권 일수": st.column_config.NumberColumn("10위권 일수", format="%.1f일"),
    "롱런 지수": st.column_config.NumberColumn("롱런 지수", format="%.2f배"),
})
st.caption("평균은 로그를 취하기 전의 원래 단위입니다. 롱런 지수는 누적 관객을 첫 주 관객으로 나눈 값이고 20에서 잘랐습니다.")

st.subheader("묶음마다 관객이 많은 다섯 편")
열 = st.columns(k)
for i, 이름 in enumerate(기호[:k]):
    묶음 = df[df["묶음"] == 이름].sort_values("누적 관객", ascending=False).head(5)
    열[i].markdown(f"**{이름} 묶음 ({len(df[df['묶음'] == 이름]):,}편)**")
    열[i].dataframe(pd.DataFrame({"영화": 묶음["movieNm"].to_numpy(),
                                "누적 관객": 묶음["누적 관객"].to_numpy()}),
                   width="stretch", hide_index=True,
                   column_config={"누적 관객": st.column_config.NumberColumn("누적 관객", format="%,d명")})
