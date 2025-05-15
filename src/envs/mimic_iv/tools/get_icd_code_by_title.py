import json
from typing import Any, Dict

from pydantic import BaseModel, Field
from sqlalchemy.engine import Engine


class RetrieveICDCodeByTitle(BaseModel):
    engine: Engine = Field(
        ...,
        description="Retrieve an ICD code from d_icd_procedures or d_icd_diagnoses by matching a title/subtitle.",
    )

    class Config:
        arbitrary_types_allowed = True

    def invoke(self, title_substring: str = "", table: str = "d_icd_procedures") -> str:
        """
        Input:
            title_substring: e.g., “percutaneous abdominal drainage”
            table: either "d_icd_procedures" or "d_icd_diagnoses"
        Output: list of dictionaries with keys "icd_code", "long_title", and "table"
        """
        from rapidfuzz import fuzz
        from sqlalchemy import text

        if table not in ("d_icd_procedures", "d_icd_diagnoses"):
            raise ValueError("Table must be 'd_icd_procedures' or 'd_icd_diagnoses'.")

        # Fetch all rows where long_title contains any part of the substring (broad filter)
        query = text(f"""
            SELECT icd_code, long_title
            FROM {table}
            WHERE lower(long_title) LIKE :pattern
        """)
        pattern = f"%{title_substring.lower()}%"
        with self.engine.connect() as conn:
            results = conn.execute(query, {"pattern": pattern}).fetchall()

        # Fuzzy match in Python
        matches = []
        for icd_code, long_title in results:
            score = fuzz.token_set_ratio(title_substring.lower(), long_title.lower())
            if score >= 70:  # threshold for fuzzy match, adjust as needed
                matches.append(
                    {"icd_code": icd_code, "long_title": long_title, "table": table}
                )

        return json.dumps(matches, indent=4)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_icd_code_by_title",
                "description": "Retrieve an ICD code from d_icd_procedures or d_icd_diagnoses by matching a title/subtitle.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title_substring": {
                            "type": "string",
                            "description": "The title substring, e.g., percutaneous abdominal drainage",
                        },
                        "table": {
                            "type": "string",
                            "description": "The table name, either d_icd_procedures or d_icd_diagnoses",
                        },
                    },
                    "required": ["title_substring", "table"],
                },
            },
        }
