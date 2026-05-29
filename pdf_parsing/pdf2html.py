import os
import os.path as op
import shutil

import pikepdf
from pypdf import PdfWriter, PdfReader
import subprocess
number_of_pages_to_limit = 10
upload_folder = 'temp'

def ensure_path(file_path, name):
    """ Create a subdir in the upload folder with the required name.
        Do not error if it already exists
    :param file_path: Full path of a file
    :param name: name of the subdir in the
    :return: name_path: Path to the file in the new folder
    """
    # create the path if it does not exist
    name_folder = op.join(upload_folder, name)
    if not op.exists(name_folder):
        os.makedirs(name_folder)

    # add the file name
    _, file = os.path.split(file_path)
    name_path = op.join(name_folder, file)
    return name_path


def split_files(file_path, temp_dir):
    def perform_split(file_path, temp_dir):
        dir_name = op.abspath(temp_dir)
        try:
            os.makedirs(dir_name)
        except FileExistsError:
            # remove files and subfolders from the directory if it already exists
            file_list = os.listdir(dir_name)
            print(file_list)
            for file_name in file_list:
                path = os.path.join(dir_name, file_name)
                if os.path.isfile(path):
                    os.remove(path)
                if os.path.isdir(path) and path.endswith('_images'):
                    shutil.rmtree(path)
        count = 0
        with open(file_path, "rb") as f:
            inputpdf = PdfReader(f)
            split_files = []
            for i in range(len(inputpdf.pages)):
                output = PdfWriter()
                output.add_page(inputpdf.pages[i])
                file_exists = True
                while file_exists:
                    count += 1
                    split_file = op.join(dir_name, f'{count}.pdf')
                    file_exists = op.isfile(split_file)

                split_files.append(split_file)
                with open(split_file, "wb") as outputStream:
                    output.write(outputStream)
        return split_files

    try:
        split_paths = perform_split(file_path, temp_dir)
    except Exception:  # Changed from specific exceptions since PyPDF has different error types
        print('conversion using pikepdf')
        file, ext = op.splitext(file_path)
        base_name = op.basename(file_path)
        pike_path = op.join('temp', base_name)
        temp = pikepdf.open(file_path)
        temp.save(pike_path)
        temp.close()
        split_paths = perform_split(pike_path, temp_dir)
    return split_paths

def pdf2html(split_paths):
    html_paths = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for split_path in split_paths:
        html_path = op.splitext(split_path)[0] + ".html"
        subprocess.run(['../../pdftron/pdf2html', '-in', f'{split_path}', '-out', f'{html_path}'], cwd=script_dir)
        html_paths.append(html_path)
    return html_paths
