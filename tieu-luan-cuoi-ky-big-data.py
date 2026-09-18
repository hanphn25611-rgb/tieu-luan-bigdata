# -*- coding: utf-8 -*-
"""
Amazon Review – Analysis Dashboard
Tiểu luận cuối kỳ Big Data
"""
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Review Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { color-scheme: light; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e8ecf0;
}
[data-testid="stSidebar"] * { color: #1a1f2e !important; }

[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Space Grotesk', sans-serif;
    color: #7c3aed !important;
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.kpi-card {
    background: #ffffff;
    border: 1px solid #e8ecf0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8a9bb0;
    margin-bottom: 0.3rem;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #1a1f2e;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #aab5c2;
    margin-top: 0.2rem;
}
.section-eyebrow {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #7c3aed;
    margin-bottom: 0.25rem;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #1a1f2e;
    margin-bottom: 1rem;
    border-bottom: 2px solid #f0f0f4;
    padding-bottom: 0.4rem;
}
.priority-badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.card-p0 {
    background: #fef2f2;
    border: 1.5px solid #fca5a5;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.card-p1 {
    background: #fffbeb;
    border: 1.5px solid #fcd34d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.card-meta {
    font-size: 0.8rem;
    color: #555;
    line-height: 1.8;
}
.card-ai {
    background: #f8f9fa;
    border-left: 3px solid #7c3aed;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.9rem;
    font-size: 0.82rem;
    font-style: italic;
    color: #444;
    margin: 0.6rem 0;
}
.card-action {
    background: #f0fdf4;
    border-left: 3px solid #16a34a;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.9rem;
    font-size: 0.82rem;
    color: #166534;
    margin-top: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT SAMPLE DATA
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_PRIORITY = pd.DataFrame([
    {"priority":"P0","aspect":"plot_and_storyline","aspect_vi":"Cốt truyện & Tình tiết","priority_score":0.031200,"n_total":312,"n_neg":187,"neg_rate":0.5994,"avg_rating":2.84,"coverage":0.1248,"intensity":0.4189,"confidence":1.0},
    {"priority":"P0","aspect":"writing_style","aspect_vi":"Văn phong & Lối viết","priority_score":0.024500,"n_total":280,"n_neg":154,"neg_rate":0.5500,"avg_rating":2.96,"coverage":0.1120,"intensity":0.3980,"confidence":1.0},
    {"priority":"P1","aspect":"character_development","aspect_vi":"Phát triển nhân vật","priority_score":0.012800,"n_total":198,"n_neg":98,"neg_rate":0.4949,"avg_rating":3.12,"coverage":0.0792,"intensity":0.3250,"confidence":1.0},
    {"priority":"P1","aspect":"pacing","aspect_vi":"Nhịp độ truyện","priority_score":0.009100,"n_total":165,"n_neg":74,"neg_rate":0.4485,"avg_rating":3.24,"coverage":0.0660,"intensity":0.3070,"confidence":1.0},
    {"priority":"P1","aspect":"value_for_price","aspect_vi":"Giá trị / Giá tiền","priority_score":0.007600,"n_total":145,"n_neg":61,"neg_rate":0.4207,"avg_rating":3.31,"coverage":0.0580,"intensity":0.3120,"confidence":1.0},
    {"priority":"P2","aspect":"ending","aspect_vi":"Phần kết truyện","priority_score":0.003200,"n_total":120,"n_neg":38,"neg_rate":0.3167,"avg_rating":3.50,"coverage":0.0480,"intensity":0.2110,"confidence":1.0},
    {"priority":"P2","aspect":"kindle_and_ebook","aspect_vi":"Kindle & Định dạng eBook","priority_score":0.002800,"n_total":98,"n_neg":29,"neg_rate":0.2959,"avg_rating":3.62,"coverage":0.0392,"intensity":0.2420,"confidence":1.0},
    {"priority":"P2","aspect":"translation_and_edition","aspect_vi":"Bản dịch & Ấn bản","priority_score":0.002100,"n_total":87,"n_neg":21,"neg_rate":0.2414,"avg_rating":3.71,"coverage":0.0348,"intensity":0.2500,"confidence":1.0},
    {"priority":"P2","aspect":"content_accuracy","aspect_vi":"Độ chính xác nội dung","priority_score":0.001500,"n_total":74,"n_neg":15,"neg_rate":0.2027,"avg_rating":3.84,"coverage":0.0296,"intensity":0.2500,"confidence":1.0},
    {"priority":"P2","aspect":"delivery_and_condition","aspect_vi":"Giao hàng & Tình trạng sách","priority_score":0.000900,"n_total":62,"n_neg":9,"neg_rate":0.1452,"avg_rating":3.92,"coverage":0.0248,"intensity":0.2500,"confidence":1.0},
])

DEFAULT_EVIDENCE = pd.DataFrame([
    {"aspect_vi":"Cốt truyện & Tình tiết","sentiment":"Negative","sentiment_score":0.95,"rating":2,"year":2023,"sentence":"The plot was predictable from the very first chapter and offered no surprises."},
    {"aspect_vi":"Cốt truyện & Tình tiết","sentiment":"Positive","sentiment_score":0.91,"rating":5,"year":2024,"sentence":"The storyline was captivating and kept me turning pages until 3am."},
    {"aspect_vi":"Văn phong & Lối viết","sentiment":"Negative","sentiment_score":0.88,"rating":2,"year":2023,"sentence":"The writing felt clunky and repetitive, making it hard to stay engaged."},
    {"aspect_vi":"Văn phong & Lối viết","sentiment":"Positive","sentiment_score":0.93,"rating":5,"year":2024,"sentence":"Beautiful prose with vivid descriptions that transported me to another world."},
    {"aspect_vi":"Phát triển nhân vật","sentiment":"Negative","sentiment_score":0.87,"rating":3,"year":2023,"sentence":"The characters felt flat and I couldn't connect with the protagonist at all."},
    {"aspect_vi":"Nhịp độ truyện","sentiment":"Negative","sentiment_score":0.82,"rating":2,"year":2022,"sentence":"The middle section dragged on for far too long with unnecessary subplots."},
    {"aspect_vi":"Giá trị / Giá tiền","sentiment":"Negative","sentiment_score":0.79,"rating":3,"year":2024,"sentence":"Not worth the price for such a thin volume with little original content."},
    {"aspect_vi":"Phần kết truyện","sentiment":"Negative","sentiment_score":0.84,"rating":2,"year":2023,"sentence":"The ending felt rushed and left too many storylines unresolved."},
    {"aspect_vi":"Kindle & Định dạng eBook","sentiment":"Negative","sentiment_score":0.76,"rating":3,"year":2023,"sentence":"The Kindle version had serious formatting issues with broken paragraphs."},
    {"aspect_vi":"Cốt truyện & Tình tiết","sentiment":"Neutral","sentiment_score":0.61,"rating":3,"year":2022,"sentence":"The story was okay, nothing spectacular but not terrible either."},
])

DEFAULT_CARDS = [
    {"aspect":"plot_and_storyline","aspect_vi":"Cốt truyện & Tình tiết","priority":"P0","priority_score":0.031200,"n_negative":187,"neg_rate":"59.9%","avg_rating":2.84,"dept":"Ban Biên tập","action":"Rà soát cấu trúc cốt truyện, tham khảo phản hồi beta readers trước khi xuất bản","ai_summary":"Customers frequently complain about predictable plots with no surprising twists. The storyline lacks originality and fails to engage readers beyond the first few chapters."},
    {"aspect":"writing_style","aspect_vi":"Văn phong & Lối viết","priority":"P0","priority_score":0.024500,"n_negative":154,"neg_rate":"55.0%","avg_rating":2.96,"dept":"Ban Biên tập","action":"Tăng cường biên tập văn phong, đảm bảo văn xuôi mạch lạc và phù hợp đối tượng đọc giả","ai_summary":"Reviewers consistently highlight clunky, repetitive writing that breaks immersion. Poor sentence structure and awkward phrasing are the most cited issues."},
    {"aspect":"character_development","aspect_vi":"Phát triển nhân vật","priority":"P1","priority_score":0.012800,"n_negative":98,"neg_rate":"49.5%","avg_rating":3.12,"dept":"Ban Biên tập","action":"Yêu cầu tác giả phát triển chiều sâu nhân vật, thêm arc cảm xúc rõ ràng","ai_summary":"Characters are described as one-dimensional with little emotional depth. Readers struggle to connect with protagonists who lack believable motivations."},
    {"aspect":"pacing","aspect_vi":"Nhịp độ truyện","priority":"P1","priority_score":0.009100,"n_negative":74,"neg_rate":"44.8%","avg_rating":3.24,"dept":"Ban Biên tập","action":"Điều chỉnh nhịp độ truyện, cắt bỏ đoạn dài không cần thiết ở giữa","ai_summary":"The middle section is consistently flagged as too slow with unnecessary filler content. Readers lose interest before reaching the climax."},
    {"aspect":"value_for_price","aspect_vi":"Giá trị / Giá tiền","priority":"P1","priority_score":0.007600,"n_negative":61,"neg_rate":"42.1%","avg_rating":3.31,"dept":"Bộ phận Định giá & Marketing","action":"Xem xét lại mức giá so với độ dày và giá trị nội dung, cân nhắc bundle hoặc khuyến mãi","ai_summary":"Buyers feel the price does not match the content quality or book length. Many suggest the work should be priced lower or bundled with additional value."},
]

# ──────────────────────────────────────────────────────────────────────────────
# Default — sẽ được ghi đè sau khi load data
DATASET_LABEL = "Amazon Books"

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 📚 {DATASET_LABEL}")
    st.markdown("Review Analysis Dashboard")
    st.markdown("---")

    data_mode = st.radio(
        "Nguồn dữ liệu",
        ["📂 Dữ liệu mẫu (mặc định)", "⬆️ Upload file của bạn"],
        index=0,
    )

    f_priority = f_evidence = f_cards = f_trend = None
    if data_mode == "⬆️ Upload file của bạn":
        st.markdown("### Tải file lên")
        f_priority = st.file_uploader("priority_ranking.csv", type="csv", key="pr")
        f_evidence = st.file_uploader("evidence_table.csv",   type="csv", key="ev")
        f_cards    = st.file_uploader("decision_cards.json",  type="json", key="dc")
        f_trend    = st.file_uploader("trend_data.csv",       type="csv", key="tr")
        st.markdown("---")
        DATASET_LABEL = st.text_input("Tên dataset", value="Amazon Review", key="dataset_label",
                                      help="Tên hiển thị trên tiêu đề và biểu đồ")

    st.markdown("---")
    st.caption("Tiểu luận cuối kỳ – Big Data")

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
if data_mode == "📂 Dữ liệu mẫu (mặc định)":
    try:
        df_priority = pd.read_csv("priority_ranking.csv")
        df_evidence = pd.read_csv("evidence_table.csv")
        with open("decision_cards.json", encoding="utf-8") as _f:
            cards = json.load(_f)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_priority = DEFAULT_PRIORITY.copy()
        df_evidence = DEFAULT_EVIDENCE.copy()
        cards       = DEFAULT_CARDS

    # Load trend data
    try:
        df_trend = pd.read_csv("trend_data.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_trend = pd.DataFrame()

else:
    if f_priority is None or f_evidence is None or f_cards is None:
        st.markdown(f"""
        <div style='text-align:center; padding:5rem 2rem;'>
            <div style='font-size:3rem; margin-bottom:1rem;'>📚</div>
            <div style='font-family:Space Grotesk,sans-serif; font-size:1.5rem;
                        font-weight:700; color:#1a1f2e; margin-bottom:0.5rem;'>
                {DATASET_LABEL} Review Analysis
            </div>
            <div style='color:#8a9bb0; font-size:0.9rem; max-width:380px; margin:0 auto;'>
                Vui lòng upload đủ 3 file từ sidebar để bắt đầu.
            </div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    df_priority = pd.read_csv(f_priority)
    df_evidence = pd.read_csv(f_evidence)
    cards       = json.load(f_cards)
    df_trend    = pd.read_csv(f_trend) if f_trend is not None else pd.DataFrame()

# ── Dataset label (dynamic) ──────────────────────────────────────────────────
if data_mode == "📂 Dữ liệu mẫu (mặc định)":
    DATASET_LABEL = "Amazon Books"
else:
    DATASET_LABEL = st.session_state.get("dataset_label", "Amazon Review")

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Tổng quan",
    "📋  Bảng Evidence",
    "🃏  Decision Cards",
    "📈  Xu hướng thời gian",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – TỔNG QUAN
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-eyebrow">Phân tích ưu tiên khía cạnh</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Tổng quan kết quả phân tích Review – {DATASET_LABEL}</div>', unsafe_allow_html=True)

    # KPI row
    n_sentences = df_evidence.shape[0]
    n_aspects  = df_priority.shape[0]
    n_p0       = int((df_priority["priority"] == "P0").sum())
    neg_overall = df_evidence[df_evidence["sentiment"] == "Negative"].shape[0] / df_evidence.shape[0]

    k1, k2, k3, k4 = st.columns(4)
    for col, label, val, sub in [
        (k1, "Tổng câu phân tích", f"{n_sentences:,}", "câu từ evidence"),
        (k2, "Khía cạnh phân tích", str(n_aspects),     "aspect categories"),
        (k3, "Ưu tiên P0",          str(n_p0),           "cần xử lý ngay"),
        (k4, "Tỷ lệ Negative",      f"{neg_overall:.0%}","trên toàn evidence"),
    ]:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Priority Score bar chart
    col_a, col_b = st.columns([1.4, 1])

    with col_a:
        st.markdown('<div class="section-eyebrow">Priority Score theo khía cạnh</div>', unsafe_allow_html=True)
        P_COLOR = {"P0": "#ef4444", "P1": "#f59e0b", "P2": "#94a3b8"}
        colors  = [P_COLOR[p] for p in df_priority["priority"]]
        fig_bar = go.Figure(go.Bar(
            x=df_priority["priority_score"],
            y=df_priority["aspect_vi"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.4f}" for v in df_priority["priority_score"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Score: %{x:.6f}<extra></extra>",
        ))
        fig_bar.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Priority Score", yaxis_title="",
            height=380, margin=dict(t=10, b=10, l=10, r=60),
            font=dict(family="Inter", size=11),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-eyebrow">Phân bố sentiment</div>', unsafe_allow_html=True)
        sent_counts = df_evidence["sentiment"].value_counts()
        SENT_COLOR  = {"Positive": "#22c55e", "Neutral": "#f59e0b", "Negative": "#ef4444"}
        fig_pie = px.pie(
            values=sent_counts.values,
            names=sent_counts.index,
            color=sent_counts.index,
            color_discrete_map=SENT_COLOR,
            hole=0.42,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(
            showlegend=False, height=260,
            margin=dict(t=10, b=10, l=10, r=10),
            font=dict(family="Inter", size=12),
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('<div class="section-eyebrow" style="margin-top:0.5rem">Neg Rate theo khía cạnh (Top 5)</div>', unsafe_allow_html=True)
        top5 = df_priority.nlargest(5, "neg_rate")[["aspect_vi","neg_rate","avg_rating"]]
        top5["neg_rate"] = (top5["neg_rate"] * 100).round(1).astype(str) + "%"
        top5["avg_rating"] = top5["avg_rating"].round(2).astype(str) + " ★"
        top5.columns = ["Khía cạnh", "Neg Rate", "Avg Rating"]
        st.dataframe(top5, hide_index=True, use_container_width=True)

    # Full table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Bảng xếp hạng đầy đủ</div>', unsafe_allow_html=True)
    show_df = df_priority[["priority","aspect_vi","priority_score","n_total","n_neg","neg_rate","avg_rating","coverage","intensity"]].copy()
    show_df.columns = ["Ưu tiên","Khía cạnh","Priority Score","Tổng câu","Câu Neg","Neg Rate","Avg Rating","Coverage","Intensity"]
    show_df["Neg Rate"] = (show_df["Neg Rate"] * 100).round(1).astype(str) + "%"
    st.dataframe(
        show_df, hide_index=True, use_container_width=True,
        column_config={
            "Priority Score": st.column_config.ProgressColumn(format="%.6f", min_value=0, max_value=float(df_priority["priority_score"].max())),
            "Ưu tiên": st.column_config.TextColumn(),
        }
    )

    # Phân phối Rating
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Phân phối Rating</div>', unsafe_allow_html=True)
    rating_counts = df_evidence["rating"].value_counts().sort_index().reset_index()
    rating_counts.columns = ["rating", "count"]
    rating_counts["label"] = rating_counts["rating"].astype(int).astype(str) + " ★"
    fig_rating = go.Figure(go.Bar(
        x=rating_counts["label"],
        y=rating_counts["count"],
        marker_color=["#ef4444","#f97316","#f59e0b","#84cc16","#22c55e"],
        text=rating_counts["count"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Số câu: %{y:,}<extra></extra>",
    ))
    fig_rating.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Rating", yaxis_title="Số câu",
        height=300, margin=dict(t=10, b=10, l=10, r=20),
        font=dict(family="Inter", size=12),
    )
    st.plotly_chart(fig_rating, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – EVIDENCE TABLE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-eyebrow">Câu evidence từ review</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Bảng Evidence – Câu phân tích theo khía cạnh & Sentiment</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        aspects_list = ["Tất cả"] + sorted(df_evidence["aspect_vi"].unique().tolist())
        selected_aspect = st.selectbox("Lọc khía cạnh", aspects_list)
    with col_f2:
        sent_list = ["Tất cả", "Positive", "Neutral", "Negative"]
        selected_sent = st.selectbox("Lọc sentiment", sent_list)
    with col_f3:
        rating_range = st.slider("Rating", 1, 5, (1, 5))

    df_ev_filtered = df_evidence.copy()
    if selected_aspect != "Tất cả":
        df_ev_filtered = df_ev_filtered[df_ev_filtered["aspect_vi"] == selected_aspect]
    if selected_sent != "Tất cả":
        df_ev_filtered = df_ev_filtered[df_ev_filtered["sentiment"] == selected_sent]
    df_ev_filtered = df_ev_filtered[
        df_ev_filtered["rating"].between(rating_range[0], rating_range[1])
    ]

    st.caption(f"{len(df_ev_filtered):,} câu được hiển thị")

    st.dataframe(
        df_ev_filtered[["aspect_vi","sentiment","sentiment_score","rating","year","sentence"]].rename(columns={
            "aspect_vi": "Khía cạnh", "sentiment": "Sentiment",
            "sentiment_score": "Độ tin cậy", "rating": "Rating",
            "year": "Năm", "sentence": "Câu evidence"
        }),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Độ tin cậy": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            "Rating": st.column_config.NumberColumn(format="%d ★"),
        }
    )

    # Sentiment by aspect bar
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Phân bố sentiment theo khía cạnh</div>', unsafe_allow_html=True)
    sent_asp = (df_evidence.groupby(["aspect_vi","sentiment"])
                .size().reset_index(name="count"))
    fig_sent = px.bar(
        sent_asp, x="aspect_vi", y="count", color="sentiment",
        color_discrete_map={"Positive":"#22c55e","Neutral":"#f59e0b","Negative":"#ef4444"},
        barmode="stack",
        labels={"aspect_vi":"Khía cạnh","count":"Số câu","sentiment":"Sentiment"},
    )
    fig_sent.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        height=320, margin=dict(t=10, b=10, l=10, r=10),
        font=dict(family="Inter", size=11),
        xaxis_tickangle=-25,
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_sent, use_container_width=True)

    # Top Bigrams
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Top Bigrams trong câu Negative P0/P1</div>', unsafe_allow_html=True)

    import re
    from collections import Counter

    STOP_WORDS_APP = {
        "the","a","an","and","or","but","in","on","at","to","for","of","with",
        "it","is","was","that","this","i","my","me","we","be","are","not",
        "have","had","as","so","do","no","its","from","he","she","they","by",
        "were","has","been","more","all","would","about","up","out","when",
        "there","their","what","which","one","can","get","just","your","our",
        "his","her","if","you","said","than","also","into","book","books",
        "read","reading","story","novel","just","very","really","even","like",
        "didn","don","doesn","wasn","isn","couldn","wouldn","too","much","how",
        "well","good","bad","great","little","only","still","ever","never","felt",
    }

    p0p1_aspects = list(df_priority[df_priority["priority"].isin(["P0","P1"])]["aspect_vi"])
    neg_sents = df_evidence[
        (df_evidence["sentiment"] == "Negative") &
        (df_evidence["aspect_vi"].isin(p0p1_aspects))
    ]["sentence"].tolist()

    bigram_counter = Counter()
    for sent in neg_sents:
        words = re.sub(r"[^a-z\s]", "", sent.lower()).split()
        words = [w for w in words if w not in STOP_WORDS_APP and len(w) > 2]
        for a, b in zip(words, words[1:]):
            bigram_counter[f"{a} {b}"] += 1

    top_bigrams = bigram_counter.most_common(15)
    if top_bigrams:
        bg_labels = [x[0] for x in top_bigrams][::-1]
        bg_vals   = [x[1] for x in top_bigrams][::-1]
        fig_bg = go.Figure(go.Bar(
            x=bg_vals, y=bg_labels, orientation="h",
            marker_color="#7c3aed",
            text=bg_vals, textposition="outside",
            hovertemplate="<b>%{y}</b>: %{x} lần<extra></extra>",
        ))
        fig_bg.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Số lần xuất hiện", yaxis_title="",
            height=420, margin=dict(t=10, b=10, l=10, r=60),
            font=dict(family="Inter", size=11),
        )
        st.plotly_chart(fig_bg, use_container_width=True)
    else:
        st.info("Không đủ dữ liệu để tính bigrams.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – DECISION CARDS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-eyebrow">Hành động đề xuất</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Decision Cards – Khía cạnh ưu tiên P0 & P1</div>', unsafe_allow_html=True)

    p0_cards = [c for c in cards if c["priority"] == "P0"]
    p1_cards = [c for c in cards if c["priority"] == "P1"]

    if p0_cards:
        st.markdown("#### 🔴 P0 — Ưu tiên cao nhất (cần xử lý ngay)")
        cols_p0 = st.columns(len(p0_cards)) if len(p0_cards) > 1 else [st.container()]
        for col, card in zip(cols_p0, p0_cards):
            with col:
                st.markdown(f"""
                <div class="card-p0">
                    <div class="card-title" style="color:#991b1b;">
                        {card['aspect_vi']}
                        <span class="priority-badge" style="background:#fecaca;color:#991b1b;font-size:0.7rem;margin-left:0.4rem;">P0</span>
                    </div>
                    <div class="card-meta">
                        📊 Priority Score: <b>{card['priority_score']:.6f}</b><br>
                        💬 Câu Negative: <b>{card['n_negative']:,}</b> ({card['neg_rate']})<br>
                        ⭐ Avg Rating: <b>{card['avg_rating']:.2f}</b><br>
                        🏢 Bộ phận: <b>{card['dept']}</b>
                    </div>
                    <div class="card-ai">🤖 AI Summary: {card['ai_summary']}</div>
                    <div class="card-action">✅ Hành động: {card['action']}</div>
                </div>""", unsafe_allow_html=True)

    if p1_cards:
        st.markdown("#### 🟡 P1 — Ưu tiên trung bình")
        cols_p1 = st.columns(min(len(p1_cards), 3))
        for i, card in enumerate(p1_cards):
            with cols_p1[i % 3]:
                st.markdown(f"""
                <div class="card-p1">
                    <div class="card-title" style="color:#92400e;">
                        {card['aspect_vi']}
                        <span class="priority-badge" style="background:#fde68a;color:#92400e;font-size:0.7rem;margin-left:0.4rem;">P1</span>
                    </div>
                    <div class="card-meta">
                        📊 Priority Score: <b>{card['priority_score']:.6f}</b><br>
                        💬 Câu Negative: <b>{card['n_negative']:,}</b> ({card['neg_rate']})<br>
                        ⭐ Avg Rating: <b>{card['avg_rating']:.2f}</b><br>
                        🏢 Bộ phận: <b>{card['dept']}</b>
                    </div>
                    <div class="card-ai">🤖 AI Summary: {card['ai_summary']}</div>
                    <div class="card-action">✅ Hành động: {card['action']}</div>
                </div>""", unsafe_allow_html=True)

    # Summary table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Tóm tắt toàn bộ Decision Cards</div>', unsafe_allow_html=True)
    PRIORITY_COLOR = {"P0": "#ef4444", "P1": "#f59e0b", "P2": "#22c55e"}
    rows_html = ""
    for c in cards:
        pri = c.get("priority", "")
        dot_color = PRIORITY_COLOR.get(pri, "#94a3b8")
        neg = c.get("neg_rate", "")
        neg_str = f"{neg:.1%}" if isinstance(neg, float) else str(neg)
        avg = c.get("avg_rating", "")
        avg_str = f"{avg:.2f}" if isinstance(avg, float) else str(avg)
        score = c.get("priority_score", "")
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        action = str(c.get("action", ""))
        dept   = str(c.get("dept", ""))
        aspect = str(c.get("aspect_vi", ""))
        td = "padding:8px 12px; white-space:nowrap; border-bottom:1px solid #f0f2f5; border-right:1px solid #f0f2f5; background:#ffffff; vertical-align:middle;"
        rows_html += f"""<tr>
            <td style="{td}"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{dot_color};margin-right:6px;vertical-align:middle;"></span>{pri}</td>
            <td style="{td}">{aspect}</td>
            <td style="{td} text-align:right;">{score_str}</td>
            <td style="{td} text-align:right;">{neg_str}</td>
            <td style="{td} text-align:right;">{avg_str}</td>
            <td style="{td}">{dept}</td>
            <td style="{td} border-right:none;">{action}</td>
        </tr>"""
    th = "padding:8px 12px; text-align:left; white-space:nowrap; border-bottom:2px solid #e8ecf0; border-right:1px solid #e8ecf0; background:#f8f9fb; color:#6b7280; font-weight:600;"
    st.markdown(f"""
    <div style="overflow-x:auto; border:1px solid #e8ecf0; border-radius:10px; max-height:300px; overflow-y:auto;">
    <table style="border-collapse:collapse; font-size:0.83rem; font-family:Inter,sans-serif; min-width:100%;">
        <thead>
            <tr>
                <th style="{th}">Ưu tiên</th>
                <th style="{th}">Khía cạnh</th>
                <th style="{th} text-align:right;">Score</th>
                <th style="{th} text-align:right;">Neg Rate</th>
                <th style="{th} text-align:right;">Avg ★</th>
                <th style="{th}">Bộ phận</th>
                <th style="{th} border-right:none;">Hành động</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – XU HƯỚNG THỜI GIAN
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-eyebrow">Phân tích xu hướng</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Xu hướng Neg Rate & Avg Rating theo thời gian</div>', unsafe_allow_html=True)

    if df_trend.empty:
        st.info("Chưa có dữ liệu xu hướng. Upload file `trend_data.csv` từ notebook để xem tab này.")
        st.stop()

    # ── Bộ lọc ──────────────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        all_aspects = sorted(df_trend["aspect_vi"].unique().tolist())
        p01_default = list(df_priority[df_priority["priority"].isin(["P0","P1"])]["aspect_vi"])
        p01_default = [a for a in p01_default if a in all_aspects]
        selected_aspects = st.multiselect(
            "Chọn khía cạnh hiển thị",
            all_aspects,
            default=p01_default,
        )
    with col_f2:
        year_min = int(df_trend["year"].min())
        year_max = int(df_trend["year"].max())
        year_range = st.slider("Khoảng năm", year_min, year_max, (2005, year_max))

    if not selected_aspects:
        st.warning("Vui lòng chọn ít nhất 1 khía cạnh.")
        st.stop()

    df_t = df_trend[
        df_trend["aspect_vi"].isin(selected_aspects) &
        df_trend["year"].between(year_range[0], year_range[1])
    ]

    COLORS_T = ["#c0392b","#e67e22","#2980b9","#8e44ad","#27ae60","#16a085","#d35400","#2c3e50","#c0392b","#7f8c8d"]

    # ── Biểu đồ Neg Rate ─────────────────────────────────────────────────────
    st.markdown('<div class="section-eyebrow">Neg Rate (%) theo năm</div>', unsafe_allow_html=True)
    fig_neg = go.Figure()
    for i, asp in enumerate(selected_aspects):
        sub = df_t[df_t["aspect_vi"] == asp].sort_values("year")
        if sub.empty:
            continue
        fig_neg.add_trace(go.Scatter(
            x=sub["year"],
            y=(sub["neg_rate"] * 100).round(1),
            mode="lines+markers",
            name=asp,
            line=dict(color=COLORS_T[i % len(COLORS_T)], width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{asp}</b><br>Năm: %{{x}}<br>Neg Rate: %{{y:.1f}}%<extra></extra>",
        ))
    fig_neg.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Năm", yaxis_title="Neg Rate (%)",
        yaxis_ticksuffix="%",
        height=380,
        margin=dict(t=10, b=10, l=10, r=10),
        font=dict(family="Inter", size=11),
        legend=dict(orientation="h", y=-0.22, x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_neg, use_container_width=True)

    # ── Biểu đồ Avg Rating ───────────────────────────────────────────────────
    st.markdown('<div class="section-eyebrow">Avg Rating theo năm</div>', unsafe_allow_html=True)
    fig_rat = go.Figure()
    for i, asp in enumerate(selected_aspects):
        sub = df_t[df_t["aspect_vi"] == asp].sort_values("year")
        if sub.empty:
            continue
        fig_rat.add_trace(go.Scatter(
            x=sub["year"],
            y=sub["avg_rating"].round(2),
            mode="lines+markers",
            name=asp,
            line=dict(color=COLORS_T[i % len(COLORS_T)], width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{asp}</b><br>Năm: %{{x}}<br>Avg Rating: %{{y:.2f}}★<extra></extra>",
        ))
    fig_rat.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Năm", yaxis_title="Avg Rating (★)",
        yaxis=dict(range=[1, 5.2]),
        height=380,
        margin=dict(t=10, b=10, l=10, r=10),
        font=dict(family="Inter", size=11),
        legend=dict(orientation="h", y=-0.22, x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_rat, use_container_width=True)

    # ── Bảng nhận xét xu hướng ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Nhận xét xu hướng theo khía cạnh</div>', unsafe_allow_html=True)

    summary_rows = []
    for asp in selected_aspects:
        sub = df_trend[df_trend["aspect_vi"] == asp].sort_values("year")
        if len(sub) < 2:
            continue
        first_neg = sub.iloc[0]["neg_rate"] * 100
        last_neg  = sub.iloc[-1]["neg_rate"] * 100
        delta     = last_neg - first_neg
        trend_icon = "▲" if delta > 0 else "▼"
        trend_str  = f"{trend_icon} {abs(delta):.1f}pp"
        avg_neg    = (sub["neg_rate"] * 100).mean()
        peak_year  = int(sub.loc[sub["neg_rate"].idxmax(), "year"])
        peak_val   = sub["neg_rate"].max() * 100
        # Lấy priority của aspect này
        pri_row = df_priority[df_priority["aspect_vi"] == asp]
        pri_str = pri_row["priority"].values[0] if len(pri_row) > 0 else "—"
        summary_rows.append({
            "Tier": pri_str,
            "Khía cạnh": asp,
            "Neg Rate TB": f"{avg_neg:.1f}%",
            "Xu hướng (đầu→cuối)": trend_str,
            "Đỉnh Neg Rate": f"{peak_year} ({peak_val:.1f}%)",
        })

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)