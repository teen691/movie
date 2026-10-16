import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
FEATURES = ["스크린 수(로그)", "누적 관객(로그)", "10위권 일수", "롱런 지수"]
FEATURE_COLS = ["first_scrn_log", "total_audi_log", "days_in_top10", "long_run_index"]
FEATURE_LABELS = dict(zip(FEATURE_COLS, FEATURES))
CLUSTER_LABELS = ["㉮", "㉯", "㉰"]

st.title("🎬 영화 유형 나누기")
st.caption("영화의 흥행 지표를 바탕으로 비슷한 영화끼리 세 유형으로 묶습니다.")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    required = [
        "movieCd", "movieNm", "first_scrn", "first_week_audi",
        "total_audi", "days_in_top10"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("필요한 열이 없습니다: " + ", ".join(missing))
    for col in ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["first_scrn_log"] = np.log10(df["first_scrn"].where(df["first_scrn"] > 0))
    df["total_audi_log"] = np.log10(df["total_audi"].where(df["total_audi"] > 0))
    df["long_run_index"] = (
        df["total_audi"] / df["first_week_audi"].where(df["first_week_audi"] > 0)
    ).clip(upper=20)
    return df

try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

all_count = len(raw)
valid_mask = (
    raw[FEATURE_COLS].notna().all(axis=1)
    & (raw["first_week_audi"] > 0)
    & np.isfinite(raw[FEATURE_COLS]).all(axis=1)
)
data = raw.loc[valid_mask].copy()
st.write(f"**전체 영화 {all_count:,}편 · 분석에 사용한 영화 {len(data):,}편**")

if len(data) < 3:
    st.error("세 묶음으로 나누려면 분석 가능한 영화가 최소 3편 필요합니다.")
    st.stop()

st.subheader("1. 묶는 데 사용할 속성")
selected = st.multiselect(
    "두 개 이상 선택하세요",
    options=FEATURE_COLS,
    default=FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS[c],
)
if len(selected) < 2:
    st.warning("k-평균으로 묶으려면 속성을 두 개 이상 선택해 주세요.")
    st.stop()

scaled = StandardScaler().fit_transform(data[selected])
model = KMeans(n_clusters=3, random_state=42, n_init=10)
data["cluster_raw"] = model.fit_predict(scaled)

# Assign ㉮ to the cluster with the greatest mean total audience.
cluster_order = (
    data.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
cluster_map = {cluster_id: CLUSTER_LABELS[i] for i, cluster_id in enumerate(cluster_order)}
data["유형"] = data["cluster_raw"].map(cluster_map)

st.subheader("2. 2차원 산점도")
axis_options = selected
col_x, col_y = st.columns(2)
with col_x:
    x_axis = st.selectbox("가로축", axis_options, index=0, format_func=lambda c: FEATURE_LABELS[c])
with col_y:
    y_default = 1 if len(axis_options) > 1 else 0
    y_axis = st.selectbox("세로축", axis_options, index=y_default, format_func=lambda c: FEATURE_LABELS[c])
fig2 = px.scatter(
    data, x=x_axis, y=y_axis, color="유형", hover_name="movieNm",
    category_orders={"유형": CLUSTER_LABELS},
    labels={x_axis: FEATURE_LABELS[x_axis], y_axis: FEATURE_LABELS[y_axis], "유형": "영화 유형"},
    hover_data={"movieCd": False, "유형": True},
)
fig2.update_traces(marker={"size": 7, "opacity": 0.8})
fig2.update_layout(legend_title_text="영화 유형", margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3. 3차원 산점도")
if len(selected) < 3:
    st.info("3차원 산점도를 보려면 위에서 속성을 세 개 이상 선택해 주세요.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        x3 = st.selectbox("X축", selected, index=0, key="x3", format_func=lambda c: FEATURE_LABELS[c])
    with c2:
        y3 = st.selectbox("Y축", selected, index=1, key="y3", format_func=lambda c: FEATURE_LABELS[c])
    with c3:
        z3 = st.selectbox("Z축", selected, index=2, key="z3", format_func=lambda c: FEATURE_LABELS[c])
    fig3 = px.scatter_3d(
        data, x=x3, y=y3, z=z3, color="유형", hover_name="movieNm",
        category_orders={"유형": CLUSTER_LABELS},
        labels={
            x3: FEATURE_LABELS[x3], y3: FEATURE_LABELS[y3],
            z3: FEATURE_LABELS[z3], "유형": "영화 유형"
        },
        hover_data={"movieCd": False, "유형": True},
    )
    fig3.update_traces(marker={"size": 3, "opacity": 0.8})
    fig3.update_layout(legend_title_text="영화 유형", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("4. 유형별 요약")
summary = data.groupby("유형").agg(**{
    "편수": ("movieNm", "size"),
    "스크린_수_평균": ("first_scrn", "mean"),
    "누적_관객_평균": ("total_audi", "mean"),
    "10위권_일수_평균": ("days_in_top10", "mean"),
    "롱런_지수_평균": ("long_run_index", "mean"),
}).reindex(CLUSTER_LABELS)
summary = summary.rename(columns={
    "스크린_수_평균": "스크린 수 평균",
    "누적_관객_평균": "누적 관객 평균",
    "10위권_일수_평균": "10위권 일수 평균",
    "롱런_지수_평균": "롱런 지수 평균",
})
summary["편수"] = summary["편수"].astype(int)
summary["스크린 수 평균"] = summary["스크린 수 평균"].round(1)
summary["누적 관객 평균"] = summary["누적 관객 평균"].round(0).map(lambda x: f"{x:,.0f}")
summary["10위권 일수 평균"] = summary["10위권 일수 평균"].round(1)
summary["롱런 지수 평균"] = summary["롱런 지수 평균"].round(2)
st.dataframe(summary, use_container_width=True)

st.subheader("5. 유형별 누적 관객 상위 영화")
for cluster in CLUSTER_LABELS:
    group = data.loc[data["유형"] == cluster].sort_values("total_audi", ascending=False).head(5)
    with st.expander(f"{cluster} 유형 — 누적 관객 상위 {len(group)}편", expanded=True):
        if group.empty:
            st.write("해당 유형에 영화가 없습니다.")
        else:
            display = group[["movieNm", "total_audi"]].rename(
                columns={"movieNm": "영화 제목", "total_audi": "누적 관객"}
            ).copy()
            display["누적 관객"] = display["누적 관객"].map(lambda x: f"{x:,.0f}")
            st.dataframe(display.reset_index(drop=True), use_container_width=True, hide_index=True)
