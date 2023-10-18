import pypandoc
import os

def convert_md_to_pdf(md_files):
    for file in md_files:
        if not file.endswith(".md"):
            continue

        base_name, _ = os.path.splitext(file)
        output_path = os.path.join("pdf", f"{base_name}.pdf")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert md to pdf using pypandoc
        output = pypandoc.convert_file(file, 'pdf', outputfile=output_path)
        assert output == ""