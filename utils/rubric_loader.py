import io
import json
import os

from fastapi import UploadFile


def load_default_rubric() -> dict:
    """
    Load default rubric from pre-formatted JSON.
    Returns dict directly (no formatting needed).
    """
    try:
        json_path = "data/formatted_rubric.json"

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Default rubric not found at {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            rubric = json.load(f)

        # Validate it has required structure
        if "criteria" not in rubric:
            raise ValueError("Default rubric missing 'criteria' field")

        return rubric

    except Exception as e:
        raise ValueError(f"Error loading default rubric: {str(e)}")


def parse_uploaded_rubric(file: UploadFile) -> dict:
    """
    Parse uploaded rubric file (JSON or Excel).
    Returns dict if JSON (already formatted), or raw data if Excel (needs formatting).
    """
    try:
        content = file.file.read()

        if file.filename.endswith(".json"):
            # Direct JSON load - check if already formatted
            rubric = json.loads(content.decode("utf-8"))

            # If it has "criteria", it's already formatted
            if "criteria" in rubric:
                return rubric
            else:
                # Unstructured JSON, needs formatting
                return rubric

        elif file.filename.endswith(".xlsx"):
            # Excel needs formatting - return raw data
            import pandas as pd

            excel_file = io.BytesIO(content)
            df = pd.read_excel(excel_file, sheet_name=0)

            # Return as string for formatter agent
            return {"raw_data": df.to_string(), "needs_formatting": True}

        else:
            raise ValueError(f"Unsupported rubric format: {file.filename}")

        file.file.seek(0)

    except Exception as e:
        raise ValueError(f"Error parsing rubric: {str(e)}")
