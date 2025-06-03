import json
import os

from sqlalchemy import create_engine

from src.envs.base import Env
from src.envs.mimic_iv.tools.get_icd_code_by_title import RetrieveICDCodeByTitle
from src.envs.mimic_iv.tools.instruction_sql_search import RAG
from src.envs.mimic_iv.tools.sql_db_list_tables import SqlDbListTables
from src.envs.mimic_iv.tools.sql_db_query import SqlDbQuery
from src.envs.mimic_iv.tools.sql_db_schema import SqlDbSchema
from src.envs.mimic_iv.tools.value_substring_search import ValueSubstringSearch
from src.retrieve import Retriever, VectorDB
from src.types_utils import Task

FOLDER_PATH = os.path.dirname(__file__)


class MimicIVEnv(Env):
    def __init__(
        self,
        eval_mode: str,
        user_strategy: str,
        user_model: str,
        task_index: int,
        db_path: str = "src/envs/mimic_iv/mimic_iv.sqlite",
    ):
        assert os.path.exists(db_path), f"Database file does not exist: {db_path}"
        with open(os.path.join(FOLDER_PATH, f"{eval_mode}_data.json"), "r") as f:
            tasks = [Task(**kwargs) for kwargs in json.load(f)]
        with open(os.path.join(FOLDER_PATH, "rules.txt"), "r") as f:
            rule = f.read()
        engine = create_engine(f"sqlite:///{db_path}")
        documentation_path = "./mimic_iv_schema.json"
        dataset_path = "./data/text_sql.csv"
        vector_db = VectorDB(
            dataset_path=dataset_path,
            index_path="./data/index/text_sql.index",
            model_name="emilyalsentzer/Bio_ClinicalBERT",
        )
        if not os.path.exists(vector_db.index_path):
            vector_db.build_index()
        else:
            vector_db.load_index()
        retriever = Retriever(vector_db)
        with open(documentation_path, "r") as file:
            documentation = json.load(file)
        sql_db_list_tables = SqlDbListTables(engine=engine)
        sql_db_schema = SqlDbSchema(engine=engine, documentation=documentation)
        sql_db_query = SqlDbQuery(engine=engine)
        value_substring_search = ValueSubstringSearch(engine=engine)
        get_icd_code_by_title = RetrieveICDCodeByTitle(engine=engine)
        instruction_sql_search = RAG(retriever=retriever)

        super().__init__(
            tools=[
                sql_db_list_tables,
                sql_db_schema,
                value_substring_search,
                sql_db_query,
                get_icd_code_by_title,
                instruction_sql_search,
            ],  # type: ignore
            tasks=tasks,
            user_strategy=user_strategy,
            user_model=user_model,
            db_path=db_path,
            task_index=task_index,
            rule=rule,
        )
