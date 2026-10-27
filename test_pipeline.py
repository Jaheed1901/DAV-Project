"""
Unit & Integration Test Script for Past Question Paper Analyzer.
Validates question extraction, topic classification, repeated question detection,
difficulty trends, predictor math, and visualization generation.
"""

import sys
import os
import io
import pandas as pd

# Ensure standard UTF-8 stream output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from modules.text_cleaner import clean_text, preprocess_for_nlp, extract_keywords
from modules.question_extractor import extract_questions_from_text, extract_marks
from modules.topic_classifier import TopicClassifier
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
from utils.helpers import get_demo_question_papers, calculate_summary_metrics


def test_text_cleaner():
    print("Testing Text Cleaner...")
    sample = "Explain normalization up to BCNF with suitable example! [10 Marks]"
    cleaned = preprocess_for_nlp(sample)
    assert "normalization" in cleaned, "Protected term 'normalization' missing"
    assert "bcnf" in cleaned, "Protected term 'bcnf' missing"
    keywords = extract_keywords(sample)
    print(f"  Cleaned text: {cleaned}")
    print(f"  Keywords extracted: {keywords}")
    print("  [OK] Text Cleaner passed.")


def test_marks_extraction():
    print("Testing Marks Extraction...")
    q1 = "Explain deadlock prevention techniques. [10 Marks]"
    marks1, text1 = extract_marks(q1)
    assert marks1 == 10.0, f"Expected 10.0, got {marks1}"

    q2 = "Define Atomicity and Durability (5M)"
    marks2, text2 = extract_marks(q2)
    assert marks2 == 5.0, f"Expected 5.0, got {marks2}"

    q3 = "Design an ER diagram [5 + 5]"
    marks3, text3 = extract_marks(q3)
    assert marks3 == 10.0, f"Expected 10.0, got {marks3}"
    print(f"  Extracted marks: {marks1}, {marks2}, {marks3}")
    print("  [OK] Marks Extraction passed.")


def test_question_extractor():
    print("Testing Question Extractor...")
    raw_exam = """
    UNIVERSITY EXAM 2024
    Time: 3 Hours  Max Marks: 100

    Q1. (a) Explain normalization up to BCNF with suitable example. [10 Marks]
    Q1. (b) What is Deadlock? Explain Wait-Die and Wound-Wait schemes. [7 Marks]
    Q2. State the ACID properties of a database transaction. [5 Marks]
    """
    questions = extract_questions_from_text(raw_exam, year=2024)
    assert len(questions) >= 3, f"Expected at least 3 questions, got {len(questions)}"
    assert questions[0]["year"] == 2024
    assert questions[0]["marks"] == 10.0
    print(f"  Extracted {len(questions)} questions successfully.")
    print("  [OK] Question Extractor passed.")


def test_topic_classifier():
    print("Testing Topic Classifier...")
    classifier = TopicClassifier("DBMS (Database Management Systems)")
    t1, c1 = classifier.classify_question("Explain 1NF, 2NF, 3NF and BCNF normalization.")
    assert "Normalization" in t1, f"Expected Normalization, got {t1}"

    t2, c2 = classifier.classify_question("Describe deadlock prevention and Wait-Die protocol.")
    assert "Concurrency Control & Deadlocks" in t2, f"Expected Deadlocks, got {t2}"

    t3, c3 = classifier.classify_question("Write SQL query using INNER JOIN, GROUP BY and HAVING.")
    assert "SQL" in t3, f"Expected SQL, got {t3}"

    print(f"  Classified sample 1: '{t1}' (conf: {c1})")
    print(f"  Classified sample 2: '{t2}' (conf: {c2})")
    print(f"  Classified sample 3: '{t3}' (conf: {c3})")
    print("  [OK] Topic Classifier passed.")


def test_similarity_and_repeated_detection():
    print("Testing Repeated Question Detection...")
    demo_df = get_demo_question_papers()
    repeated = detect_repeated_questions(demo_df, similarity_threshold=0.60)
    assert len(repeated) > 0, "Expected repeated questions to be found in demo dataset"
    print(f"  Detected {len(repeated)} repeated question clusters.")
    for grp in repeated[:3]:
        print(f"  - Theme: {grp['representative_question'][:45]}... (Appeared in {grp['years']})")
    print("  [OK] Repeated Question Detection passed.")


def test_difficulty_analysis():
    print("Testing Difficulty Analysis...")
    d1, r1 = assess_question_difficulty("Define candidate key and primary key.", marks=2.0)
    assert d1 == "Easy", f"Expected Easy, got {d1}"

    d2, r2 = assess_question_difficulty("Explain 1NF, 2NF and 3NF with examples.", marks=6.0)
    assert d2 == "Medium", f"Expected Medium, got {d2}"

    d3, r3 = assess_question_difficulty("Design and construct an ER diagram for a Hospital System.", marks=10.0)
    assert d3 == "Hard", f"Expected Hard, got {d3}"

    demo_df = get_demo_question_papers()
    trends = compute_difficulty_trends(demo_df)
    assert "yearly_stats" in trends
    print(f"  Difficulty levels: '{d1}', '{d2}', '{d3}'")
    print(f"  Trend Direction: {trends['trend_direction']}")
    print("  [OK] Difficulty Analysis passed.")


def test_predictor():
    print("Testing Exam Trend Predictor...")
    demo_df = get_demo_question_papers()
    repeated = detect_repeated_questions(demo_df, similarity_threshold=0.60)
    preds = predict_upcoming_topics(demo_df, repeated_groups=repeated)
    assert len(preds) > 0, "Predictions should not be empty"
    top_pred = preds[0]
    assert "priority_score" in top_pred
    assert 0 <= top_pred["priority_score"] <= 100
    print(f"  #1 Predicted Topic: {top_pred['topic']} (Score: {top_pred['priority_score']}/100, Tier: {top_pred['priority_tier']})")
    print("  [OK] Predictor passed.")


def test_visualizations():
    print("Testing Matplotlib Visualizations (All 7 Charts)...")
    demo_df = get_demo_question_papers()
    repeated = detect_repeated_questions(demo_df, similarity_threshold=0.60)
    preds = predict_upcoming_topics(demo_df, repeated_groups=repeated)

    fig1 = plot_top_frequent_topics(demo_df)
    fig2 = plot_topic_frequency_by_year(demo_df)
    fig3 = plot_marks_distribution_by_topic(demo_df)
    fig4 = plot_difficulty_distribution_by_year(demo_df)
    fig5 = plot_difficulty_trend(demo_df)
    fig6 = plot_predicted_topics(preds)
    fig7 = plot_repeated_questions_summary(repeated)

    assert fig1 is not None and fig2 is not None and fig3 is not None
    assert fig4 is not None and fig5 is not None and fig6 is not None and fig7 is not None
    print("  [OK] All 7 charts generated successfully without errors.")


if __name__ == "__main__":
    print("=== RUNNING PAST QUESTION PAPER ANALYZER TEST SUITE ===")
    test_text_cleaner()
    test_marks_extraction()
    test_question_extractor()
    test_topic_classifier()
    test_similarity_and_repeated_detection()
    test_difficulty_analysis()
    test_predictor()
    test_visualizations()
    print("\nALL TESTS PASSED SUCCESSFULLY!")
