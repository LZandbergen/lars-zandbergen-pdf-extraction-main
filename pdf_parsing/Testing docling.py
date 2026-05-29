# %%
import docling
from docling.document_converter import DocumentConverter
import json

source =  "../data/allshoes_catalogus/pdfs/allshoes_catalogus.pdf" # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())
# Export the converted document to a JSON file
with open('docling_document.json', 'w') as json_file:
    json.dump(result.document.export_to_dict(), json_file)

# Export the converted document to a Markdown file
with open('docling_document.md', 'w') as markdown_file:
    markdown_file.write(result.document.export_to_markdown())

# %%
import importlib
import sys
import os
import markdownify
import pdf2html, scrape_html
importlib.reload(pdf2html)
importlib.reload(scrape_html)
sys.path.append("..")


# Define the file path and temporary directory
file_path = "../data/allshoes_catalogus/pdfs/allshoes_catalogus.pdf"
temp_dir = "../pdf_parsing/.temp/pdftron/shoes/"
temp_dir += file_path.split('/')[-1].replace('.pdf', '')

# Split the PDF into individual pages
split_paths = pdf2html.split_files(file_path, temp_dir)

# Convert the split PDF pages to HTML
html_paths = pdf2html.pdf2html(split_paths)

# Scrape the HTML files to extract data into a pandas dataframe
new_df = scrape_html.scrape_html(html_paths)

# Convert the dataframe to a txt file and save it
new_df.to_csv(f'{temp_dir}/pdftron.txt', sep='\t', index=False)

# Convert HTML files to Markdown and combine them
markdown_content = ''
for html_file in html_paths:
    with open(html_file, 'r', encoding='utf-8') as f:
        html = f.read()
    # Convert HTML to Markdown with headers and tables
    md = markdownify.markdownify(html, heading_style="ATX")
    markdown_content += md + '\n\n'

# Save the combined markdown content to a file
with open(f'{temp_dir}/PDFtron.md', 'w', encoding='utf-8') as md_file:
    md_file.write(markdown_content)

print('Markdown file generated from HTML files.')

print(new_df.head())
print('done')


# %%



