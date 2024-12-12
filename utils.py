from fpdf import FPDF
import logging

def save_qa_history_pdf(qa_history):
    """Savol-javob tarixini PDF formatida saqlaydi."""
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Savol-Javob Tarixi', ln=True, align='C')
        pdf.ln(10)

        for qa in qa_history:
            pdf.set_font('Arial', 'B', 12)
            pdf.multi_cell(0, 10, f"Savol: {qa['Savol']}")
            pdf.set_font('Arial', '', 12)
            pdf.multi_cell(0, 10, f"Javob: {qa['Javob']}")
            pdf.ln(5)

        pdf_path = "qa_history.pdf"
        pdf.output(pdf_path)
        return pdf_path
    except Exception as e:
        logging.error(f"PDF yaratishda xatolik: {e}")
        return None
