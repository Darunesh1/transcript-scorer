import io
import json

import pandas as pd
from fastapi import UploadFile


def load_default_rubric() -> str:
    """Load default rubric from Excel file."""
    try:
        df = pd.read_excel("data/default_rubric.xlsx", sheet_name="Rubrics")
        return df.to_string()
    except Exception as e:
        raise ValueError(f"Error loading default rubric: {str(e)}")


def parse_uploaded_rubric(file: UploadFile) -> str:
    """Parse uploaded rubric (Excel or JSON)."""
    try:
        content = file.file.read()

        if file.filename.endswith(".xlsx"):
            excel_file = io.BytesIO(content)
            df = pd.read_excel(excel_file, sheet_name=0)
            rubric_str = df.to_string()

        elif file.filename.endswith(".json"):
            rubric_dict = json.loads(content.decode("utf-8"))
            rubric_str = json.dumps(rubric_dict, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file.filename}")

        file.file.seek(0)
        return rubric_str

    except Exception as e:
        raise ValueError(f"Error parsing rubric: {str(e)}")
