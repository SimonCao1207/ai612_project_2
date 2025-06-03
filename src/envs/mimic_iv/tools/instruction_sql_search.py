from typing import Any, Dict

from pydantic import BaseModel, Field

from src.retrieve import Retriever


class RAG(BaseModel):
    retriever: Retriever = Field(
        ..., description="The retriever to extract relevant samples."
    )

    class Config:
        arbitrary_types_allowed = True

    def invoke(self, user_query: str = "") -> str:
        results = self.retriever.retrieve(user_query)
        if not results:
            return "No similar samples found."
        formatted = []
        for sample, _ in results:
            question = sample.get("question", "")
            sql = sample.get("sql", "")
            formatted.append(f"\nQuestion: {question}\nSQL: {sql}")
        return "\n\n".join(formatted)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "instruction_sql_search",
                "description": "extract samples similar to the given instruction for reference",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {
                            "type": "string",
                            "description": "User query",
                        }
                    },
                    "required": [],
                },
            },
        }
