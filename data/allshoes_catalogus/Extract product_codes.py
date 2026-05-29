# %%
import pdfplumber
import re

def extract_product_codes(pdf_path):
    """Extracts unique five-digit product codes from a PDF based on the 'art.nr.' pattern."""
    try:
        product_codes = set()
        pattern = r'art\.nr\.\s(\d{5})'  # Regex to capture five-digit codes after "art.nr."

        # Open the PDF using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:  # Ensure text was extracted
                    matches = re.findall(pattern, text)
                    product_codes.update(matches)

        return list(product_codes)

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Usage example:
pdf_path = 'pdfs/allshoes_catalogus.pdf'
product_codes = extract_product_codes(pdf_path)
for code in product_codes:
    print(code)

# %%




