from typing import Any, Dict

from pydantic import BaseModel, Field
from sqlalchemy import inspect
from sqlalchemy.engine import Engine


class SqlDbListTables(BaseModel):
    engine: Engine = Field(..., description="The engine to list tables from.")

    class Config:
        arbitrary_types_allowed = True

    def invoke(self, tool_input: str = "") -> str:
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        return ", ".join(tables)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "sql_db_list_tables",
                "description": "Get the list of table names in the database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_input": {
                            "type": "string",
                            "description": "An empty string; no input required.",
                        }
                    },
                    "required": [],
                },
            },
        }
