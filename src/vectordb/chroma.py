import json

import chromadb
from chromadb.utils import embedding_functions

from vectordb.base import VectorStore


class ChromaDB(VectorStore):
    # TODO create ChromaDB for vector store
    def __init__(self, config: dict):
        self._config = config
        self._client = chromadb.PersistentClient(path="schema_db")
        self._embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    def generate_embedding(self, data: str, **kwargs) -> list[float]:
        pass

    def add_documentation(
        self,
        documentation: str,
        metadatas=None,
        ids=None**kwargs,
    ) -> str:
        pass

    def return_table_docs(self, list_tables: list[str]):
        pass


if __name__ == "__main__":
    # Demo

    doc = json.load(open("mimic_iv_schema.json"))
    table_desc = []
    table_metas = []
    table_ids = []
    column_desc = []
    column_metas = []
    column_ids = []
    links_texts = {}
    for module in doc.keys():
        schema = doc[module]
        for table, info in schema["tables"].items():
            links = "; ".join(
                f"Table {table} links to table {other_table} on columns {', '.join(on_column)}"
                for other_table, on_column in info["links to"].items()
            )
            if links:
                links_texts[table] = links
            table_desc.append(f"Table {table}: {info['description']}")
            table_metas.append({"table": table})
            table_ids.append(table)
            for col, desc in info["columns"].items():
                column_desc.append(f"{table}.{col}: {desc}")
                column_metas.append(
                    {
                        "table": table,
                        "column": col,
                    }
                )
                column_ids.append(f"{table}.{col}")

    # print("Table descriptions:", table_desc[0])
    # print("Column descriptions:", column_desc[0])
    # exit()

    client = chromadb.PersistentClient(path="schema_db")
    embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    table_collection = client.get_or_create_collection(
        name="table_collection", embedding_function=embed_func
    )
    column_collection = client.get_or_create_collection(
        name="column_collection", embedding_function=embed_func
    )

    # Add table documents
    table_collection.add(ids=table_ids, documents=table_desc, metadatas=table_metas)

    # Add column documents
    column_collection.add(ids=column_ids, documents=column_desc, metadatas=column_metas)

    user_query = "I need to check if patient ID 10014729 had a resection surgery. Can you help me with that?"

    query_results = table_collection.query(
        query_texts=[user_query],
        n_results=3,
    )
    print(query_results)

    print("---------------------")

    query_results = column_collection.query(
        query_texts=[user_query],
        n_results=3,
    )
    print(query_results)
