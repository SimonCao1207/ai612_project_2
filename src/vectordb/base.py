from abc import ABC, abstractmethod


class VectorStore(ABC):
    def __init__(self, config: dict):
        self._config = config

    @abstractmethod
    async def generate_embedding(self, data: str, **kwargs) -> list[float]:
        pass

    @abstractmethod
    async def add_documentation(
        self,
        documentation: str,
        question: str = None,
        metadatas=None,
        add_name=None,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    async def return_table_docs(self, list_tables: list[str]):
        """
        Get documentation and metadata for a list of tables.

        Args:
            list_tables (list): List of table names

        Returns:
            tuple: (documentation string, list of table names)
        """
        pass
