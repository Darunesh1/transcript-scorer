import io

import PyPDF2
from fastapi import UploadFile


def extract_text_from_file(file: UploadFile) -> str:
    """Extract text from uploaded TXT or PDF file."""
    try:
        content = file.file.read()

        if file.filename.endswith(".txt"):
            text = content.decode("utf-8")

        elif file.filename.endswith(".pdf"):
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        else:
            raise ValueError(f"Unsupported file type: {file.filename}")

        file.file.seek(0)
        return text.strip()

    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")
