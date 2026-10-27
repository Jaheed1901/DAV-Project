# 🎓 Past Question Paper Analyzer & Exam Trend Predictor

An intelligent, explainable Python application designed to analyze multi-year university and board examination question papers—both **digitally typed PDFs** and **scanned papers/images**—to extract high-value academic insights, detect question repetition, track difficulty trends, and predict high-priority topics for upcoming examinations.

---

## 📌 1. Problem Statement & Motivation

Preparing for semester or board examinations often involves sifting through five or ten years of unorganized question papers. Students face several challenges:
- **Time Inefficiency:** Manually cross-referencing questions to identify important chapters consumes valuable revision time.
- **Subjective Guesswork:** Students rely on rumors or vague intuitions rather than objective historical trends to judge what is important.
- **Scanned & Inaccessible Papers:** University archives often provide poor-quality scanned PDFs or photographed question sheets rather than clean digital text.
- **Equal Weight Fallacy:** A topic appearing in a 2-mark definition is often given the same attention as one appearing in a 10-mark design problem.

### 💡 Objective
This project bridges this gap by creating an end-to-end, automated, and explainable examination analytics pipeline:
1. Ingests PDFs, images, and scanned papers via a dual OCR + direct PDF extraction engine.
2. Extracts individual questions, identifies sub-parts, and detects marks.
3. Classifies questions into core subject topics using keyword rules and TF-IDF Cosine Similarity.
4. Detects repeated questions (exact duplicates, rephrased queries, and semantic equivalents).
5. Assesses question cognitive difficulty using Bloom's Taxonomy.
6. Employs a multi-factor scoring model to estimate high-priority topics for the next exam.
7. Presents all insights through an interactive Streamlit dashboard with publication-quality Matplotlib visualizations.

---

## 🚀 2. Core Features

- **Dual-Engine Ingestion (OCR + PDF Parsing):** Automatically detects whether a PDF is digitally typed or scanned. Text-based PDFs are parsed natively via `pdfplumber`/`pypdf`, while scanned PDFs and images (`.png`, `.jpg`) pass through OpenCV preprocessing (grayscale, Otsu thresholding, Gaussian denoise, deskew) and Tesseract OCR.
- **Intelligent Question Delimiter & Marks Detection:** Heuristic regex engine detects standard examination formats (`Q1.`, `1.(a)`, `(i)`, etc.) and extracts mark brackets (`[10 Marks]`, `(5M)`, `[5+5]`).
- **Semantic Topic Classification:** Combines domain keyword dictionaries with TF-IDF vectorization and Cosine Similarity so that differently worded questions (e.g. *"Explain normalization"* vs *"Describe 1NF, 2NF, 3NF and BCNF"*) are mapped into the same broader topic.
- **Cross-Year Repeated Question Detection:** Clusters questions across examination sessions based on string normalization, token n-grams, and Cosine Similarity thresholds.
- **Bloom's Taxonomy Difficulty Estimator:** Categorizes questions into **Easy**, **Medium**, or **Hard** using cognitive action verbs (*Define*, *Explain*, *Analyze*, *Design*) paired with mark weights. Computes yearly difficulty indexes to identify whether papers are becoming easier, harder, or remaining stable.
- **Multi-Factor Exam Trend Predictor:** Estimates upcoming exam priorities using an explainable 4-factor scoring model:
  - **Frequency Score (30%)**
  - **Marks Weightage (30%)**
  - **Recency Decay Score (15%)**
  - **Repetition Bonus (25%)**
- **Rich Visual Dashboard:** Interactive Streamlit dashboard with 7 comprehensive Matplotlib analytical charts, search filters, and one-click CSV export.

---

## 🛠️ 3. Technologies Used

| Domain | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core backend and algorithms |
| **User Interface** | Streamlit | Web dashboard, interactive filters, metrics |
| **OCR & Vision** | Tesseract OCR, Pytesseract, OpenCV, Pillow | Scanned paper preprocessing, deskewing, text recognition |
| **PDF Extraction**| pdfplumber, pypdf | Direct text and image extraction from typed PDFs |
| **NLP & Text** | NLTK, Scikit-learn | Tokenization, lemmatization, stop words, TF-IDF vectorizer |
| **Analytics** | Pandas, NumPy | Data transformations, cross-tabulation, aggregation |
| **Visualization**| Matplotlib | Custom-styled charts and analytical graphs |

---

## 🏗️ 4. System Architecture

```text
Question Papers (PDFs / Scanned Images / Photos)
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│               Ingestion & Preprocessing                │
├───────────────────────────┬────────────────────────────┤
│   Text-Based PDF Flow     │    Scanned / Image Flow    │
│  (pdfplumber / pypdf)     │   (OpenCV Preprocessing)   │
│   Extracts native text    │   Grayscale → Denoise      │
│                           │   → Otsu Thresh → Deskew   │
│                           │   → Pytesseract OCR        │
└───────────────────────────┴────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│           Text Cleaning & NLP Preprocessing            │
│  • Strip examination boilerplate (Time, Max Marks)     │
│  • Lowercasing & Tokenization                          │
│  • Domain Protected Terms preservation (DBMS, BCNF)    │
│  • Stop word filtering & WordNet Lemmatization         │
└────────────────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│          Question Boundary & Marks Extraction          │
│  • Regex patterns: Q1(a), 1., (i), Section A           │
│  • Marks regex: [10 Marks], (5M), [5+5=10]             │
└────────────────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│      Topic Classification (Hybrid TF-IDF Model)        │
│  • Domain keyword dictionary mapping                   │
│  • TF-IDF Cosine Similarity matching against topics    │
└────────────────────────────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│ Similarity &    │ │ Cognitive       │
│ Repeated        │ │ Difficulty      │
│ Questions       │ │ Analysis        │
│ (TF-IDF Cosine) │ │ (Bloom's Verbs) │
└─────────────────┘ └─────────────────┘
         │                   │
         └─────────┬─────────┘
                   ▼
┌────────────────────────────────────────────────────────┐
│       High-Weightage Topic Prediction Engine           │
│  Topic Score = 0.30·Freq + 0.30·Marks +               │
│                0.15·Recency + 0.25·Repetition          │
│  Normalized to 0–100 Scale (Very High, High, Med, Low) │
└────────────────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│            Interactive Streamlit Dashboard             │
│  • 7 Matplotlib Charts    • Semantic Question Search   │
│  • CSV Dataset Export     • Customizable Weights       │
└────────────────────────────────────────────────────────┘
```

---

## 📁 5. Project Structure

```text
question-paper-analyzer/
│
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Project dependencies
├── README.md                   # Complete documentation & viva guide
├── sample_data_generator.py    # Generates 5-year demo examination dataset
├── processed_questions.csv     # Structured processed question dataset
│
├── data/
│   ├── raw/                    # Raw question papers (.pdf, .txt, .png)
│   └── processed/              # Processed CSVs and reports
│
├── modules/
│   ├── __init__.py
│   ├── ocr.py                  # OpenCV image preprocessing & Tesseract OCR
│   ├── pdf_parser.py           # Native PDF parsing & OCR routing
│   ├── text_cleaner.py         # NLTK tokenization, cleaning, lemmatization
│   ├── question_extractor.py   # Question boundary & marks regex extraction
│   ├── topic_classifier.py     # Keyword & TF-IDF Cosine topic classification
│   ├── similarity.py           # Cross-year repeated question clustering
│   ├── difficulty.py           # Bloom's taxonomy difficulty & yearly trend
│   ├── predictor.py            # Multi-factor high-weightage topic predictor
│   └── visualization.py        # All 7 Matplotlib visualization figures
│
└── utils/
    ├── __init__.py
    └── helpers.py              # File validators, metrics, sample dataset
```

---

## ⚙️ 6. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14.
- Optional for Scanned Documents: [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki). *(Typed PDFs run completely without Tesseract).*

### Step 1: Clone or Open Project Directory
```bash
cd "c:\Users\Khushi Ghodki\Desktop\DAV project"
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

*(If using Windows and Tesseract is installed in a non-standard directory, you can specify its path directly in the sidebar of the Streamlit dashboard).*

### Step 3: Populate Demo Dataset
To initialize the demo data (5 years of DBMS examination papers from 2021 to 2025):
```bash
python sample_data_generator.py
```

### Step 4: Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 7. Visual Charts Included

The dashboard includes all 7 analytical charts requested:
1. **Chart 1: Top Most Frequently Asked Topics** (Horizontal bar chart with exact count labels).
2. **Chart 2: Topic Frequency by Year** (Grouped multi-year comparison).
3. **Chart 3: Marks Distribution Across Topics** (Total marks accumulated per topic).
4. **Chart 4: Difficulty Distribution by Year** (100% stacked bar chart showing Easy %, Medium %, Hard %).
5. **Chart 5: Difficulty Trend Over Time** (Composite Difficulty Index curve with trend indicator).
6. **Chart 6: Top Predicted Topics for Next Exam** (Ranked priority bars color-coded by priority tier).
7. **Chart 7: Repeated-Question Analysis** (Recurrence count and years appeared for repeated themes).

---

## 🧮 8. Explainable Prediction Scoring Formula

Unlike black-box models, the priority estimation uses a clear, explainable formula:

$$\text{Priority Score} = \left( w_f \cdot \tilde{F} + w_m \cdot \tilde{M} + w_r \cdot \tilde{R} + w_p \cdot \tilde{P} \right) \times 100$$

Where:
- $\tilde{F}$: Min-Max normalized question frequency count across all years.
- $\tilde{M}$: Min-Max normalized total marks accumulated by the topic.
- $\tilde{R}$: Recency factor (questions from recent years receive progressively higher weights: $2025 \to 1.0, 2024 \to 0.8, 2023 \to 0.6$).
- $\tilde{P}$: Repetition score (bonus points awarded based on whether questions in this topic appeared in multiple years).
- $w_f = 0.30, w_m = 0.30, w_r = 0.15, w_p = 0.25$ (customizable via interactive sliders in the dashboard).

### Priority Tiers:
- 🔥 **80 – 100:** Very High Priority (Essential core topic, recurring marks)
- ⚠️ **60 – 79:** High Priority (Strong historical presence)
- 📚 **40 – 59:** Medium Priority (Standard foundational concepts)
- 💡 **0 – 39:** Low Priority (Infrequently asked questions)

---

## 🎓 9. College Presentation & Viva Guide

### Key Questions & Answers for Viva:

1. **Q: Why combine Keyword Matching with TF-IDF instead of using a deep learning LLM?**
   - *Answer:* Real-world university question papers have distinct technical vocabularies (e.g., "BCNF", "Wait-Die", "LRU"). Deep neural networks require expensive GPU resources, training data, and are non-explainable. The hybrid approach provides sub-second execution, zero server costs, and complete transparency where examiners and students can trace exactly why a question was assigned to a topic.

2. **Q: How does the system handle scanned papers vs typed PDFs?**
   - *Answer:* The parser first checks if digital text streams exist using `pdfplumber`. If valid text with more than 100 characters is found, it bypasses OCR completely for 100% precision and high speed. If the PDF contains raster images or empty text streams, it extracts the page images, runs OpenCV filters (Gaussian blur, Otsu binarization, deskewing), and feeds them to Tesseract OCR.

3. **Q: How are repeated questions detected across different years?**
   - *Answer:* The system uses a two-tier similarity filter:
     - Normalization check for identical question texts.
     - Character and word n-gram TF-IDF vectorization with Cosine Similarity ($S_c \ge 0.62$). Questions within the same topic require a lower threshold, while cross-topic comparisons use a stricter threshold.

4. **Q: How does Bloom's Taxonomy determine difficulty?**
   - *Answer:* Cognitive verbs dictate the Bloom level:
     - Level 1 & 2 (Remember/Understand): *Define*, *List*, *State* $\to$ Easy.
     - Level 3 & 4 (Apply/Explain): *Explain*, *Describe*, *Compare* $\to$ Medium.
     - Level 5 & 6 (Analyze/Create): *Analyze*, *Design*, *Derive*, *Solve* $\to$ Hard.
     Combined with mark weighting ($\le 3$ marks $\to$ Easy, $\ge 8$ marks $\to$ Hard), this produces an explainable classification.

---

## 🔮 10. Future Improvements
- Multi-language question paper recognition (Hindi, Marathi, regional state board exams).
- Question diagram detection via YOLO/OpenCV bounding box detectors.
- Automated model answer generation with step-by-step marking rubrics.
- Direct syllabus integration to flag out-of-syllabus questions.

---

## 📜 License
Developed for educational, university, and academic project evaluation purposes. Free to use and modify.
