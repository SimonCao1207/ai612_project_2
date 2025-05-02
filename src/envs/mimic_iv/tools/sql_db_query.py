from typing import Any, Dict

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


class SqlDbQuery(BaseModel):
    engine: Engine = Field(..., description="The engine to execute queries on.")

    class Config:
        arbitrary_types_allowed = True

    def invoke(self, query: str, k: int = 100) -> str:
        result = ""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                result = result.fetchall()
            n = len(result)
            base_response = str(result[:k])
            if n > k:
                additional = n - k
                base_response += f"\n\nNote: There are {additional} results not shown (out of {n} total results)."
        except SQLAlchemyError as e:
            """Format the error message"""
            base_response = f"Error: {e}"
        return base_response

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "sql_db_query",
                "description": "Execute a SQL query against the database and get back the result. If the query is not correct, an error message will be returned. The maximum number of results to return is 100.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A valid SQL query to execute.",
                        },
                        "k": {
                            "type": "integer",
                            "description": "The maximum number of results to return. Default is 100.",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
