from typing import Any, Dict

from pydantic import BaseModel, Field
from sqlalchemy.engine import Engine


class SearchDoc(BaseModel):
    engine: Engine = Field(
        ..., description="The engine to get description about the database tables."
    )

    class Config:
        arbitrary_types_allowed = True

    def invoke(self, user_query: str = "") -> str:
        pass

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_doc",
                "description": "",
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
