from typing import Any, Dict

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.engine import Engine


class SqlDbSchema(BaseModel):
    engine: Engine = Field(
        ..., description="The engine to retrieve schema and sample rows for."
    )
    documentation: dict = Field(
        ...,
        description="The documentation of the database",
    )

    class Config:
        arbitrary_types_allowed = True

    def get_table_description(self, table: str = "d_icd_procedures") -> dict:
        if table.lower() == "cost":
            return {}
        for module in self.documentation.values():
            for table_name, table_info in module["tables"].items():
                if table_name == table:
                    return {
                        "description": table_info.get(
                            "description", "No description available."
                        ).strip(),
                        "columns": table_info.get("columns", {}),
                        "links_to": table_info.get("links_to", []),
                    }
        return {}

    def invoke(self, table_names: str) -> str:
        result = []
        # Split the comma-separated table names and iterate over them
        for table in table_names.split(","):
            table = table.strip()
            table_description = self.get_table_description(table)
            if table_description:
                result.append(
                    f"- Table {table} description:\n{table_description['description']}"
                )
            try:
                with self.engine.connect() as conn:
                    # Fetch table schema
                    query_schema = f"PRAGMA table_info({table});"
                    schema_rows = conn.execute(text(query_schema)).fetchall()
                    schema = [tuple(row) for row in schema_rows] if schema_rows else []

                    # Fetch foreign keys
                    query_foreign_keys = f"PRAGMA foreign_key_list({table});"
                    fk_rows = conn.execute(text(query_foreign_keys)).fetchall()
                    foreign_keys = [tuple(row) for row in fk_rows] if fk_rows else []

                    # Fetch unique constraints
                    query_unique_keys = f"PRAGMA index_list({table});"
                    unique_rows = conn.execute(text(query_unique_keys)).fetchall()
                    unique_keys_list = (
                        [tuple(row) for row in unique_rows] if unique_rows else []
                    )
                    # PRAGMA index_list returns: (seq, name, unique, origin, partial)
                    # We're filtering for those with origin 'u'
                    unique_keys = [uk[1] for uk in unique_keys_list if uk[3] == "u"]
                    unique_keys_only = []
                    for key in unique_keys:
                        query = f"PRAGMA index_info('{key}');"
                        index_info_rows = conn.execute(text(query)).fetchall()
                        if index_info_rows:
                            index_info = [tuple(row) for row in index_info_rows]
                            unique_keys_only.append(
                                index_info[0][2]
                            )  # third element is the column name

                    # Fetch sample rows (limit to 3)
                    query_sample = f"SELECT * FROM {table} LIMIT 3;"
                    sample_rows = conn.execute(text(query_sample)).fetchall()
                    samples = [tuple(row) for row in sample_rows] if sample_rows else []

                # Build schema string for the table
                schema_str = f"- Table {table} schema\nCREATE TABLE {table} ("
                columns_list = []
                for col in schema:
                    # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
                    col_name = col[1]
                    col_type = (
                        col[2].upper().replace("INT", "INTEGER") if col[2] else ""
                    )
                    if "TIMESTAMP" in col_type:
                        col_type = "TIMESTAMP"
                    not_null = "NOT NULL" if col[3] else ""
                    columns_list.append(
                        f"\n\t{col_name} {col_type} {not_null}".rstrip()
                    )
                schema_str += ",".join(columns_list)

                # Add primary keys if defined
                primary_keys = [col[1] for col in schema if col[5]]
                if primary_keys:
                    schema_str += f",\n\tPRIMARY KEY ({', '.join(primary_keys)})"

                # Add foreign keys
                for fk in foreign_keys:
                    # PRAGMA foreign_key_list returns: (id, seq, table, from, to, on_update, on_delete, match)
                    schema_str += (
                        f",\n\tFOREIGN KEY ({fk[3]}) REFERENCES {fk[2]} ({fk[4]})"
                    )

                # Add unique constraints
                for unique_col in unique_keys_only:
                    schema_str += f",\n\tUNIQUE ({unique_col})"

                schema_str = schema_str.rstrip(",\n") + "\n)"

                # Build sample rows string
                column_names = [col[1] for col in schema]
                sample_rows_str = (
                    f"\n- 3 rows from {table} table:\n" + "\t".join(column_names) + "\n"
                )
                sample_rows_str += (
                    "\n".join(["\t".join(map(str, row)) for row in samples]) + "\n"
                )

                # Add column descriptions
                if table_description:
                    column_descriptions_str = (
                        f"\n- Descriptions of table {table}'s columns:\n"
                    )
                    for col in column_names:
                        column_description = table_description["columns"].get(col, None)
                        if column_description is None:
                            continue
                        column_descriptions_str += f"\t{col}: {column_description}\n"
                    result.append(
                        schema_str + sample_rows_str + column_descriptions_str
                    )
                else:
                    result.append(schema_str + sample_rows_str)
            except Exception:
                result.append(f"Error: table_names {{'{table}'}} not found in database")
        return "\n\n\n".join(result)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        """
        Provides metadata about the tool.

        Returns:
            Dict[str, Any]: A dictionary containing the tool's name, description, and parameters.
        """
        return {
            "type": "function",
            "function": {
                "name": "sql_db_schema",
                "description": "Retrieve the description, schema, column details, and sample rows for one or more specified tables.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_names": {
                            "type": "string",
                            "description": "A comma-separated list of table names to retrieve schema and sample rows for.",
                        }
                    },
                    "required": ["table_names"],
                },
            },
        }
