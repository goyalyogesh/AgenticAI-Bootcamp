"""
Generate dummy PDF documents for the document extraction demo.

Creates:
1. financial_report.pdf - Financial report with table (PyPDF2 mangles, pdfplumber preserves)
2. multi_column_report.pdf - Two-column layout (optional, for multi-column demo)

Run from class_demo directory: python test_documents/generate_demo_pdf.py
"""

import os
from pathlib import Path

# Use reportlab if available; otherwise fall back to creating via fpdf2 or skip
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        BaseDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Frame,
        PageTemplate,
        FrameBreak,
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


SCRIPT_DIR = Path(__file__).resolve().parent


def create_financial_report_pdf():
    """Create a financial report PDF with a table - same content that PyPDF2 mangles."""
    if not HAS_REPORTLAB:
        print("ReportLab not installed. Install with: pip install reportlab")
        return None

    path = SCRIPT_DIR / "financial_report.pdf"
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=12,
    )
    story.append(Paragraph("Fiscal Year 2024 Financial Report", title_style))
    story.append(Spacer(1, 12))

    # Section header
    story.append(Paragraph("Revenue Growth", styles["Heading2"]))
    story.append(Spacer(1, 6))

    # Table: Quarter | Revenue
    table_data = [
        ["Quarter", "Revenue"],
        ["Q1", "$2.1M"],
        ["Q2", "$2.8M"],
        ["Q3", "$3.4M"],
        ["Q4", "$4.2M"],
    ]
    t = Table(table_data, colWidths=[2 * inch, 2 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTSIZE", (0, 1), (-1, -1), 10),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 16))

    # Key metrics
    story.append(Paragraph("Key Metrics:", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("• Customer Acquisition: 78%", styles["Normal"]))
    story.append(Paragraph("• Annual Growth: 23%", styles["Normal"]))
    story.append(Paragraph("• Customer Retention: 92%", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Executive summary
    story.append(
        Paragraph(
            "Executive Summary: This report covers the financial performance for fiscal year 2024. "
            "Q4 revenue reached $4.2M, representing a 23% increase from Q3.",
            styles["Normal"],
        )
    )

    doc.build(story)
    print(f"Created: {path}")
    return path


def create_multi_column_pdf():
    """Create a true two-column PDF (academic style) for multi-column extraction demo.

    Uses BaseDocTemplate with two Frames side-by-side. Content flows:
    - Left column first (top to bottom), then
    - Right column (top to bottom).

    PyPDF2 often reads in visual line order (left line 1, right line 1, ...) = wrong.
    pdfplumber with layout can read left column fully, then right column = correct.
    """
    if not HAS_REPORTLAB:
        return None

    path = SCRIPT_DIR / "multi_column_report.pdf"
    margin = 54
    page_w, page_h = letter
    col_width = (page_w - 2 * margin) / 2
    frame_height = page_h - 2 * margin

    # Two frames side-by-side: left column, then right column
    frame_left = Frame(
        margin,
        margin,
        col_width - 6,  # small gap between columns
        frame_height,
        id="left",
        showBoundary=0,
    )
    frame_right = Frame(
        margin + col_width + 6,
        margin,
        col_width - 6,
        frame_height,
        id="right",
        showBoundary=0,
    )

    doc = BaseDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
    )
    doc.addPageTemplates(
        PageTemplate(id="twocol", frames=[frame_left, frame_right])
    )

    styles = getSampleStyleSheet()
    story = []

    # ---- LEFT COLUMN ----
    story.append(Paragraph("Research: Machine Learning Applications", styles["Heading1"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Abstract", styles["Heading2"]))
    story.append(
        Paragraph(
            "This study examines the effects of machine learning algorithms on business decision-making. "
            "We analyzed data from 200 companies over a 3-year period.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 12))
    story.append(Paragraph("Introduction", styles["Heading2"]))
    story.append(
        Paragraph(
            "The left column contains the introduction. Machine learning has transformed how businesses operate. "
            "Prior research has shown significant improvements in efficiency when ML systems are properly implemented. "
            "This paragraph continues in the left column so that the column has enough text to fill visibly.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Academic papers and newspapers often use two-column layouts. "
            "Text flows from the top of the left column to the bottom, then from the top of the right column to the bottom.",
            styles["Normal"],
        )
    )

    # Switch to RIGHT column
    story.append(FrameBreak())

    # ---- RIGHT COLUMN ----
    story.append(Paragraph("Methods and Results", styles["Heading2"]))
    story.append(
        Paragraph(
            "In contrast, the right column contains different content. "
            "Multi-column PDFs cause PyPDF2 to read line-by-line across columns (wrong order): "
            "left line 1, right line 1, back to left line 2, and so on.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "pdfplumber with layout detection reads the left column fully, then the right column (correct order). "
            "We conducted a comprehensive survey. Data was collected through interviews and questionnaires.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 12))
    story.append(Paragraph("Conclusion", styles["Heading2"]))
    story.append(
        Paragraph(
            "Results showed 35% improvement in decision-making speed when ML systems were used. "
            "Layout-aware extraction is essential for multi-column documents.",
            styles["Normal"],
        )
    )

    doc.build(story)
    print(f"Created: {path} (two-column layout)")
    return path


def ensure_demo_pdfs_exist():
    """Create demo PDFs if they don't exist. Returns path to financial_report.pdf."""
    financial_path = SCRIPT_DIR / "financial_report.pdf"
    if not financial_path.exists() and HAS_REPORTLAB:
        create_financial_report_pdf()
    if not financial_path.exists():
        raise FileNotFoundError(
            "financial_report.pdf not found. Run: pip install reportlab && python test_documents/generate_demo_pdf.py"
        )
    return financial_path


if __name__ == "__main__":
    print("Generating demo PDFs...")
    create_financial_report_pdf()
    create_multi_column_pdf()
    print("Done.")
