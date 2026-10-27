"""
Utility to generate valid sample PDF question papers for testing uploads.
"""

import os


def create_minimal_pdf(filename: str, title: str, questions: list):
    """
    Constructs a valid PDF 1.4 file containing the examination text.
    """
    stream_lines = [
        "BT",
        "/F1 16 Tf",
        "50 750 Td",
        f"({title}) Tj",
        "/F1 10 Tf",
        "0 -25 Td",
        "(Time: 3 Hours                  Max Marks: 100) Tj",
        "0 -20 Td",
        "(Note: Answer all questions.) Tj",
        "0 -25 Td"
    ]

    for q in questions:
        safe_q = q.replace("(", "\\(").replace(")", "\\)")
        stream_lines.append(f"({safe_q}) Tj")
        stream_lines.append("0 -20 Td")

    stream_lines.append("ET")
    stream_content = "\n".join(stream_lines)
    stream_len = len(stream_content)

    pdf_body = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length {stream_len} >>
stream
{stream_content}
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000300 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
500
%%EOF
"""
    with open(filename, "wb") as f:
        f.write(pdf_body.encode("latin-1"))


def generate_all_sample_pdfs():
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)

    papers = {
        "DBMS_2024_Examination.pdf": (
            "UNIVERSITY SEMESTER EXAMINATION 2024 - DBMS",
            [
                "Q1. Describe 1NF, 2NF, 3NF and BCNF with relational anomalies. [10 Marks]",
                "Q2. Explain Two-Phase Locking protocol and prevent concurrency conflicts. [8 Marks]",
                "Q3. Analyze conflict serializability using precedence graph. [10 Marks]",
                "Q4. Design an ER diagram for an E-commerce shopping portal. [10 Marks]",
                "Q5. Construct B+ Tree of order 3 for given numerical keys. [10 Marks]",
                "Q6. What is Deadlock? Explain Wait-Die and Wound-Wait schemes. [7 Marks]",
                "Q7. Write SQL queries demonstrating INNER JOIN and subqueries. [7 Marks]",
                "Q8. Define candidate key, primary key and foreign key. [4 Marks]"
            ]
        ),
        "DBMS_2025_Examination.pdf": (
            "UNIVERSITY SEMESTER EXAMINATION 2025 - DBMS",
            [
                "Q1. (a) Explain normalization up to BCNF with suitable example. [10 Marks]",
                "Q1. (b) Describe different techniques used for preventing deadlock. [8 Marks]",
                "Q2. (a) Explain ACID properties of database transactions. [7 Marks]",
                "Q2. (b) Design an ER diagram for a Bank Management System. [10 Marks]",
                "Q3. (a) Solve and verify conflict serializability using precedence graph. [10 Marks]",
                "Q3. (b) Differentiate between B-Tree and B+ Tree indexing. [7 Marks]",
                "Q4. (a) Write SQL queries using GROUP BY and HAVING clauses. [8 Marks]",
                "Q4. (b) List any four disadvantages of unnormalized database tables. [2 Marks]"
            ]
        )
    }

    for fname, (title, qs) in papers.items():
        create_minimal_pdf(os.path.join("data", "raw", fname), title, qs)
        print(f"Generated sample PDF: data/raw/{fname}")


if __name__ == "__main__":
    generate_all_sample_pdfs()
