"""
Sample Data Generator Script.
Populates mock examination question papers and initial processed dataset
so the application can be demonstrated immediately out-of-the-box.
"""

import os
from utils.helpers import get_demo_question_papers, save_processed_questions, generate_sample_raw_files


def main():
    print("Generating demo question papers and sample dataset...")
    # 1. Generate sample raw paper files in data/raw/
    generate_sample_raw_files()
    print(" Sample raw papers generated in 'data/raw/'")

    # 2. Generate structured demo questions dataset
    demo_df = get_demo_question_papers()
    saved_root = save_processed_questions(demo_df, "processed_questions.csv")
    if saved_root:
        print(" Saved processed questions to 'processed_questions.csv' and 'data/processed/'")
    print(f" Total sample questions generated: {len(demo_df)}")
    print(f" Years covered: {sorted(demo_df['year'].unique())}")


if __name__ == "__main__":
    main()
