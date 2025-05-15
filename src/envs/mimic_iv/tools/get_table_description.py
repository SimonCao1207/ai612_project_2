import json
from typing import Any, Dict

from pydantic import BaseModel, Field


class TableDescription(BaseModel):
    documentation: dict = Field(
        ...,
        description="The documentation of the database",
    )

    @classmethod
    def from_file(cls, documentation_path: str):
        with open(documentation_path, "r") as file:
            documentation = json.load(file)
        return cls(documentation=documentation)

    class Config:
        arbitrary_types_allowed = True

    def invoke(self, table: str = "d_icd_procedures") -> str:
        """
        Input:
            table: The table name
        Output: table description and its columns.
        """
        for _, module in self.documentation.items():
            for table_name, table_info in module["tables"].items():
                if table_name == table:
                    description = table_info.get(
                        "description", "No description available."
                    )
                    columns = table_info.get("columns", [])
                    return json.dumps(
                        {
                            "description": description,
                            "columns": columns,
                            "links_to": table_info.get("links_to", []),
                        },
                        indent=4,
                    )
        return "Table not found."

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_table_description",
                "description": "Get the description of a table in the database, including its columns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "the table name",
                        },
                    },
                    "required": ["table"],
                },
            },
        }
