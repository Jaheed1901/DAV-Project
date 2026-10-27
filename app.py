"""
Streamlit Web Dashboard for
Past Question Paper Analyzer & Exam Trend Predictor.
"""

import os
import io
import time
import pandas as pd
import streamlit as st

# Application Module Imports
from modules.ocr import configure_tesseract
from modules.pdf_parser import parse_question_paper_file
from modules.question_extractor import extract_questions_from_text
from modules.topic_classifier import TopicClassifier, DEFAULT_SUBJECT_TAXONOMY
from modules.similarity import detect_repeated_questions
from modules.difficulty import assess_question_difficulty, compute_difficulty_trends
from modules.predictor import predict_upcoming_topics
from modules.visualization import (
    plot_top_frequent_topics,
    plot_topic_frequency_by_year,
    plot_marks_distribution_by_topic,
    plot_difficulty_distribution_by_year,
    plot_difficulty_trend,
    plot_predicted_topics,
    plot_repeated_questions_summary
)
from utils.helpers import (
    ensure_directories,
    save_processed_questions,
    load_processed_questions,
    calculate_summary_metrics,
    get_demo_question_papers,
    generate_sample_raw_files
)

# Page configuration
st.set_page_config(
    page_title="Past Question Paper Analyzer & Exam Trend Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern visual polish
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
    }
    .priority-badge-very-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        display: inline-block;
    }
    .priority-badge-high {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        display: inline-block;
    }
    .priority-badge-medium {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

ensure_directories()

# Initialize Session State
if "df" not in st.session_state:
    loaded = load_processed_questions("processed_questions.csv")
    if loaded is not None and not loaded.empty:
        st.session_state["df"] = loaded
    else:
        st.session_state["df"] = pd.DataFrame(columns=[
            "year", "question_number", "question_text", "marks", "topic", "difficulty"
        ])

if "repeated_groups" not in st.session_state:
    st.session_state["repeated_groups"] = []

if "predictions" not in st.session_state:
    st.session_state["predictions"] = []

if "active_subject" not in st.session_state:
    st.session_state["active_subject"] = "DBMS (Database Management Systems)"


def run_pipeline(subject_name: str, sim_threshold: float = 0.62, weights: tuple = (0.30, 0.30, 0.15, 0.25)):
    """Runs topic classification, repeated question detection, difficulty assignment, and predictions."""
    df = st.session_state["df"]
    if df.empty:
        return

    # 1. Topic Classification
    classifier = TopicClassifier(subject_name)
    questions_list = df.to_dict("records")
    classifier.classify_all(questions_list)

    # 2. Difficulty Assessment
    for q in questions_list:
        diff, _ = assess_question_difficulty(q.get("question_text", ""), float(q.get("marks", 5.0)))
        q["difficulty"] = diff

    updated_df = pd.DataFrame(questions_list)
    st.session_state["df"] = updated_df

    # 3. Repeated Questions Detection
    groups = detect_repeated_questions(updated_df, similarity_threshold=sim_threshold)
    st.session_state["repeated_groups"] = groups

    # 4. Exam Trend Prediction
    wf, wm, wr, wp = weights
    preds = predict_upcoming_topics(
        updated_df,
        repeated_groups=groups,
        weight_frequency=wf,
        weight_marks=wm,
        weight_recency=wr,
        weight_repetition=wp
    )
    st.session_state["predictions"] = preds

    # Save to disk
    save_processed_questions(updated_df, "processed_questions.csv")


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/exam.png", width=64)
    st.title("Exam Trend Analyzer")
    st.caption("AI-Powered Question Paper Insights & Trend Predictor")

    st.markdown("---")

    # Subject Selector
    available_subjects = list(DEFAULT_SUBJECT_TAXONOMY.keys())
    selected_subject = st.selectbox(
        "Select Academic Subject",
        available_subjects,
        index=0,
        help="Selects the subject ontology for keyword and TF-IDF topic mapping."
    )
    st.session_state["active_subject"] = selected_subject

    # Tesseract Status Check
    tess_ok, tess_msg = configure_tesseract()
    with st.expander("OCR Engine Status (Tesseract)", expanded=not tess_ok):
        if tess_ok:
            st.success("Tesseract OCR Detected & Ready")
        else:
            st.warning("Tesseract not detected in standard paths.")
            st.info("Typed PDFs work without OCR. For scanned papers, specify tesseract.exe path below:")
            custom_tess = st.text_input("Custom Tesseract Path", placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
            if custom_tess:
                c_ok, c_msg = configure_tesseract(custom_tess)
                if c_ok:
                    st.success("Tesseract path updated!")
                    tess_ok = True
                else:
                    st.error(c_msg)

    st.markdown("---")
    st.subheader("📁 Upload Question Papers")
    uploaded_files = st.file_uploader(
        "Upload Past Papers (PDF, PNG, JPG, TXT)",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
        accept_multiple_files=True,
        help="Upload past exam papers. The year will be inferred from the filename or header."
    )

    manual_year_override = st.number_input(
        "Default / Fallback Year",
        min_value=2000,
        max_value=2035,
        value=2025,
        step=1,
        help="Used if year cannot be detected from filename or text."
    )

    # Process Button
    if uploaded_files:
        if st.button("🚀 Process Uploaded Papers", type="primary", use_container_width=True):
            with st.spinner("Extracting text and identifying questions..."):
                all_new_questions = []
                progress_bar = st.progress(0)

                for idx, file_item in enumerate(uploaded_files):
                    fname = file_item.name
                    # Read text or run OCR
                    if fname.endswith(".txt"):
                        content = file_item.read().decode("utf-8", errors="ignore")
                        extracted = {
                            "filename": fname,
                            "year": manual_year_override,
                            "raw_text": content,
                            "status": "success"
                        }
                    else:
                        extracted = parse_question_paper_file(
                            file_item,
                            filename=fname,
                            manual_year=manual_year_override
                        )

                    if extracted.get("raw_text"):
                        q_items = extract_questions_from_text(
                            extracted["raw_text"],
                            year=extracted.get("year", manual_year_override)
                        )
                        all_new_questions.extend(q_items)
                        st.sidebar.caption(f"✓ {fname}: Extracted {len(q_items)} questions (Year {extracted.get('year')})")
                    else:
                        st.sidebar.error(f"✕ {fname}: {extracted.get('message', 'No text extracted')}")

                    progress_bar.progress((idx + 1) / len(uploaded_files))

                if all_new_questions:
                    new_df = pd.DataFrame(all_new_questions)
                    # Merge with existing or overwrite
                    if not st.session_state["df"].empty:
                        combined = pd.concat([st.session_state["df"], new_df], ignore_index=True)
                        # Drop exact duplicates if re-uploaded
                        combined = combined.drop_duplicates(subset=["year", "question_number", "question_text"])
                        st.session_state["df"] = combined
                    else:
                        st.session_state["df"] = new_df

                    run_pipeline(selected_subject)
                    st.success(f"Successfully processed {len(all_new_questions)} questions!")
                    st.rerun()
                else:
                    st.error("No questions could be extracted from the uploaded files.")

    st.markdown("---")
    # Quick Demo Dataset Loader
    st.subheader("💡 Demo Mode")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if st.button("⚡ Load Demo (5 Yrs)", use_container_width=True, help="Load 5 years of sample DBMS question papers (2021-2025)"):
            with st.spinner("Loading demo dataset..."):
                demo_df = get_demo_question_papers()
                generate_sample_raw_files()
                st.session_state["df"] = demo_df
                run_pipeline(selected_subject)
                st.rerun()

    with col_d2:
        if st.button("🗑️ Reset All", use_container_width=True):
            st.session_state["df"] = pd.DataFrame(columns=[
                "year", "question_number", "question_text", "marks", "topic", "difficulty"
            ])
            st.session_state["repeated_groups"] = []
            st.session_state["predictions"] = []
            if os.path.exists("processed_questions.csv"):
                try:
                    os.remove("processed_questions.csv")
                except Exception:
                    pass
            st.rerun()


# --- MAIN DASHBOARD CONTENT ---
st.markdown('<div class="main-title">Past Question Paper Analyzer & Exam Trend Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent NLP analysis of multi-year examination papers, marks-weightage patterns, repeated questions, and upcoming topic prediction.</div>', unsafe_allow_html=True)

df = st.session_state.get("df", pd.DataFrame())
repeated_groups = st.session_state.get("repeated_groups", [])
predictions = st.session_state.get("predictions", [])

# If dataset is loaded, ensure predictions are synced
if not df.empty and (not predictions or not repeated_groups):
    run_pipeline(st.session_state["active_subject"])
    repeated_groups = st.session_state.get("repeated_groups", [])
    predictions = st.session_state.get("predictions", [])

# KPI METRIC CARDS
metrics = calculate_summary_metrics(df, repeated_groups)
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{metrics['total_papers']}</div>
        <div class="metric-label">Exam Papers ({metrics['years_span']})</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{metrics['total_questions']}</div>
        <div class="metric-label">Total Questions</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{metrics['total_topics']}</div>
        <div class="metric-label">Topics Identified</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{metrics['repeated_questions_count']}</div>
        <div class="metric-label">Repeated Questions</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{metrics['total_marks']}</div>
        <div class="metric-label">Total Analyzed Marks</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if df.empty:
    st.info("👋 Welcome! Upload question papers via the sidebar or click **'⚡ Load Demo (5 Yrs)'** to immediately explore the system.")
    st.stop()


# DASHBOARD TABS
tab_overview, tab_frequency, tab_predictions, tab_repeats, tab_difficulty, tab_search = st.tabs([
    "📊 Overview & Question Bank",
    "📈 Topic & Marks Analytics",
    "🎯 Exam Trend Predictor",
    "🔁 Repeated Questions",
    "📚 Difficulty Analysis",
    "🔍 Question Search"
])


# ==========================================================
# TAB 1: OVERVIEW & QUESTION BANK
# ==========================================================
with tab_overview:
    st.subheader("Structured Question Repository")
    st.caption("Extracted and classified questions with marks and difficulty estimations.")

    # Filter controls
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 2, 1.5, 2])
    with f_col1:
        years_avail = ["All"] + sorted([str(y) for y in df["year"].unique()])
        sel_year = st.selectbox("Filter by Year", years_avail)
    with f_col2:
        topics_avail = ["All"] + sorted(df["topic"].dropna().unique().tolist())
        sel_topic = st.selectbox("Filter by Topic", topics_avail)
    with f_col3:
        diff_avail = ["All"] + sorted(df["difficulty"].dropna().unique().tolist())
        sel_diff = st.selectbox("Filter by Difficulty", diff_avail)
    with f_col4:
        search_kw = st.text_input("Search Text", placeholder="e.g. normalization, deadlock")

    # Apply filters
    filtered_df = df.copy()
    if sel_year != "All":
        filtered_df = filtered_df[filtered_df["year"] == int(sel_year)]
    if sel_topic != "All":
        filtered_df = filtered_df[filtered_df["topic"] == sel_topic]
    if sel_diff != "All":
        filtered_df = filtered_df[filtered_df["difficulty"] == sel_diff]
    if search_kw:
        filtered_df = filtered_df[filtered_df["question_text"].str.contains(search_kw, case=False, na=False)]

    st.dataframe(
        filtered_df[[
            "year", "question_number", "question_text", "marks", "topic", "difficulty"
        ]],
        use_container_width=True,
        height=380
    )

    # Export Download Button
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Full Dataset (CSV)",
        data=csv_buffer.getvalue(),
        file_name="processed_questions.csv",
        mime="text/csv",
        help="Export all extracted questions with assigned topics and marks."
    )


# ==========================================================
# TAB 2: TOPIC FREQUENCY & WEIGHTED IMPORTANCE
# ==========================================================
with tab_frequency:
    st.subheader("Topic Frequency & Marks-Weighted Distribution")
    st.caption("Analyze which chapters dominate exams historically and carry the highest marks.")

    col_t1, col_t2 = st.columns([3, 2])

    with col_t1:
        st.pyplot(plot_top_frequent_topics(df), use_container_width=True)

    with col_t2:
        st.markdown("#### Topic Breakdown Summary")
        topic_summary = df.groupby("topic").agg(
            Count=("question_number", "count"),
            Total_Marks=("marks", "sum"),
            Avg_Marks=("marks", "mean")
        ).sort_values(by="Total_Marks", ascending=False).reset_index()

        total_marks_all = topic_summary["Total_Marks"].sum()
        topic_summary["% Marks Share"] = ((topic_summary["Total_Marks"] / total_marks_all) * 100).round(1).astype(str) + "%"
        topic_summary["Weighted Importance"] = (topic_summary["Count"] * topic_summary["Avg_Marks"]).round(1)
        topic_summary["Avg_Marks"] = topic_summary["Avg_Marks"].round(1)

        st.dataframe(
            topic_summary[["topic", "Count", "Total_Marks", "Avg_Marks", "% Marks Share", "Weighted Importance"]],
            use_container_width=True,
            height=300
        )

    st.markdown("---")
    col_c2, col_c3 = st.columns(2)
    with col_c2:
        st.pyplot(plot_topic_frequency_by_year(df), use_container_width=True)
    with col_c3:
        st.pyplot(plot_marks_distribution_by_topic(df), use_container_width=True)


# ==========================================================
# TAB 3: EXAM TREND PREDICTOR
# ==========================================================
with tab_predictions:
    st.subheader("🎯 High-Weightage Topic Prediction for Upcoming Examination")
    st.info(
        "**Explainable Priority Model:** Ranks topics based on 4 historical factors: "
        "Historical Frequency (30%), Marks Allocation (30%), Recency Weight (15%), and Question Repetition (25%). "
        "Scores are normalized on a 0–100 scale to help students prioritize revision."
    )

    # Interactive Weights Expander
    with st.expander("⚙️ Customize Priority Scoring Weights", expanded=False):
        col_w1, col_w2, col_w3, col_w4 = st.columns(4)
        with col_w1:
            w_freq = st.slider("Frequency Weight", 0.0, 1.0, 0.30, 0.05)
        with col_w2:
            w_marks = st.slider("Marks Weight", 0.0, 1.0, 0.30, 0.05)
        with col_w3:
            w_rec = st.slider("Recency Weight", 0.0, 1.0, 0.15, 0.05)
        with col_w4:
            w_rep = st.slider("Repetition Weight", 0.0, 1.0, 0.25, 0.05)

        total_w = w_freq + w_marks + w_rec + w_rep
        if total_w != 1.0:
            st.caption(f"Weights sum to {total_w:.2f}. They will be automatically normalized.")
            w_freq, w_marks, w_rec, w_rep = w_freq/total_w, w_marks/total_w, w_rec/total_w, w_rep/total_w

        if st.button("Apply Adjusted Weights"):
            run_pipeline(
                st.session_state["active_subject"],
                weights=(w_freq, w_marks, w_rec, w_rep)
            )
            st.rerun()

    # Display Top Predictions Chart
    st.pyplot(plot_predicted_topics(predictions), use_container_width=True)

    st.markdown("### Topic Priority Recommendations")

    # Group predictions by tiers
    very_high = [p for p in predictions if p["priority_score"] >= 80]
    high = [p for p in predictions if 60 <= p["priority_score"] < 80]
    medium = [p for p in predictions if 40 <= p["priority_score"] < 60]
    low = [p for p in predictions if p["priority_score"] < 40]

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        if very_high:
            st.markdown("#### 🔥 VERY HIGH PRIORITY (Score 80–100)")
            for p in very_high:
                with st.container():
                    st.markdown(f"""
                    **{p['rank']}. {p['topic']}** — <span class="priority-badge-very-high">Score: {p['priority_score']}/100</span>
                    - **Frequency:** {p['frequency']} appearances | **Avg Marks:** {p['avg_marks']}M | **Total Marks:** {p['total_marks']}M
                    - **Appeared in:** {', '.join(str(y) for y in p['years_appeared'])}
                    - **Repeated Questions Detected:** {p['repetition_count']} occurrences
                    - *Strategy:* {p['actionable_tip']}
                    """, unsafe_allow_html=True)
                    st.markdown("---")

        if high:
            st.markdown("#### ⚠️ HIGH PRIORITY (Score 60–79)")
            for p in high:
                st.markdown(f"""
                **{p['rank']}. {p['topic']}** — <span class="priority-badge-high">Score: {p['priority_score']}/100</span>
                - **Frequency:** {p['frequency']} appearances | **Avg Marks:** {p['avg_marks']}M
                - **Appeared in:** {', '.join(str(y) for y in p['years_appeared'])}
                - *Strategy:* {p['actionable_tip']}
                """, unsafe_allow_html=True)
                st.markdown("---")

    with col_p2:
        if medium:
            st.markdown("#### 📚 MEDIUM PRIORITY (Score 40–59)")
            for p in medium:
                st.markdown(f"""
                **{p['rank']}. {p['topic']}** — <span class="priority-badge-medium">Score: {p['priority_score']}/100</span>
                - **Frequency:** {p['frequency']} appearances | **Avg Marks:** {p['avg_marks']}M
                - *Strategy:* {p['actionable_tip']}
                """, unsafe_allow_html=True)
                st.markdown("---")

        if low:
            st.markdown("#### 💡 LOW PRIORITY (Score 0–39)")
            for p in low:
                st.markdown(f"- **{p['topic']}** (Score: {p['priority_score']}/100) — {p['actionable_tip']}")


# ==========================================================
# TAB 4: REPEATED QUESTIONS DETECTOR
# ==========================================================
with tab_repeats:
    st.subheader("🔁 Cross-Year Repeated Question Analysis")
    st.caption("Identifies exact, slightly rephrased, and semantically identical questions appearing across different years.")

    st.pyplot(plot_repeated_questions_summary(repeated_groups), use_container_width=True)

    if repeated_groups:
        st.markdown(f"**Found {len(repeated_groups)} repeated question themes across examination years:**")

        for idx, grp in enumerate(repeated_groups):
            years_str = ", ".join(str(y) for y in grp["years"])
            is_cross = "⭐ Multi-Year Recurrence" if grp["is_cross_year"] else "Single Session Repetition"

            with st.expander(f"Group {grp['group_id']}: {grp['topic']} — Repeated {grp['frequency']} times across {years_str} ({is_cross})"):
                st.markdown(f"**Representative Concept:** {grp['representative_question']}")
                st.markdown(f"**Average Marks:** {grp['avg_marks']} Marks | **Cosine Similarity:** {grp['avg_similarity']}")

                st.markdown("##### Variations Across Papers:")
                for q_item in grp["questions"]:
                    st.markdown(f"- **[{q_item['year']}] {q_item.get('question_number', 'Q')}** ({q_item.get('marks', 5.0)} Marks): *\"{q_item['question_text']}\"*")
    else:
        st.info("No repeated questions detected with the current similarity threshold.")


# ==========================================================
# TAB 5: DIFFICULTY TRENDS
# ==========================================================
with tab_difficulty:
    st.subheader("📚 Examination Difficulty Trend Analysis")
    st.caption("Assesses questions into Easy, Medium, and Hard using Bloom's Taxonomy, action verbs, and marks weight.")

    diff_analysis = compute_difficulty_trends(df)

    # Trend Direction Indicator
    trend_dir = diff_analysis.get("trend_direction", "Stable")
    if trend_dir == "Harder":
        st.error(f"📈 **Trend: Becoming Harder** — {diff_analysis.get('trend_explanation')}")
    elif trend_dir == "Easier":
        st.success(f"📉 **Trend: Becoming Easier** — {diff_analysis.get('trend_explanation')}")
    else:
        st.info(f"⚖️ **Trend: Stable Difficulty** — {diff_analysis.get('trend_explanation')}")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.pyplot(plot_difficulty_distribution_by_year(df), use_container_width=True)
    with col_d2:
        st.pyplot(plot_difficulty_trend(df), use_container_width=True)

    st.markdown("#### Yearly Difficulty Distribution Table")
    if not diff_analysis["yearly_stats"].empty:
        st.dataframe(diff_analysis["yearly_stats"], use_container_width=True)


# ==========================================================
# TAB 6: SEMANTIC QUESTION SEARCH
# ==========================================================
with tab_search:
    st.subheader("🔍 Interactive Question Search & Knowledge Retrieval")
    st.caption("Search across past papers by keyword or concept to find all historical question variations.")

    user_query = st.text_input("Enter concept or question keyword", placeholder="e.g., deadlock, 2PL, B+ tree, ACID, Normalization")

    if user_query:
        query_lower = user_query.lower()
        matched = df[df["question_text"].str.contains(query_lower, case=False, na=False)]

        if not matched.empty:
            st.success(f"Found {len(matched)} questions matching '{user_query}':")
            for _, row in matched.iterrows():
                st.markdown(f"""
                - **[{row['year']}] {row['question_number']}** ({row['marks']} Marks | Topic: `{row['topic']}` | Difficulty: `{row['difficulty']}`):
                  > {row['question_text']}
                """)
        else:
            st.warning(f"No questions directly found with keyword '{user_query}'. Try broader terms like 'SQL', 'tree', or 'transaction'.")
