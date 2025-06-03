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
        threshold = 0.2  # Define a threshold for similarity
        results = self.retriever.retrieve(user_query)
        filtered = [
            (sample, distance) for sample, distance in results if distance < threshold
        ]
        if not filtered:
            return "No similar samples found."
        formatted = [
            "The following question and SQL query pairs are most similar to the user query and can be used as reference:"
        ]
        for sample, _ in filtered:
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
