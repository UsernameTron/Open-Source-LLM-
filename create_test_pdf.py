from fpdf import FPDF

def create_test_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    
    test_text = """This is a test PDF with some positive content.
    The project is going great and I'm happy with the progress.
    The implementation is excellent and working well.
    Everything is fantastic and the results are amazing."""
    
    # Write text to PDF
    pdf.multi_cell(0, 10, txt=test_text)
    
    # Save the PDF
    pdf.output('test.pdf')

if __name__ == '__main__':
    create_test_pdf()
