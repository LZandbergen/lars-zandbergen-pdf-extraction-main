from . import pdf2html, scrape_html

if __name__ == "__main__":
    # Ask user for file path
    file_path = input("Enter the absolute path to the PDF file: ")
    temp_dir = input("Enter where you want to store the pdfs: ") or "/home/lzandbergen/Squafdra/lars-zandbergen-pdf-extraction/data/allshoes_catalogus/pdfs/parsed"
    split_paths = pdf2html.split_files(file_path, temp_dir)
    html_paths = pdf2html.pdf2html(split_paths)
    new_df = scrape_html.scrape_html(html_paths)
    print(new_df.head())
    print('done')