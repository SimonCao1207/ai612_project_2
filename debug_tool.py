import json
import os

from sqlalchemy import create_engine

from src.envs.mimic_iv.tools.instruction_sql_search import RAG
from src.envs.mimic_iv.tools.sql_db_schema import SqlDbSchema
from src.retrieve import Retriever, VectorDB

list_of_tables = [
    "patients",
    "admissions",
    "d_icd_diagnoses",
    "d_icd_procedures",
    "d_labitems",
    "d_items",
    "diagnoses_icd",
    "procedures_icd",
    "labevents",
    "prescriptions",
    "cost",
    "chartevents",
    "inputevents",
    "outputevents",
    "microbiologyevents",
    "icustays",
    "transfers",
]
db_path = "src/envs/mimic_iv/mimic_iv.sqlite"
engine = create_engine(f"sqlite:///{db_path}")
documentation_path = "./mimic_iv_schema.json"
with open(documentation_path, "r") as file:
    documentation = json.load(file)
sql_db_schema = SqlDbSchema(engine=engine, documentation=documentation)

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
instruction_sql_search = RAG(retriever=retriever)

# for table_name in list_of_tables:
#     print("-------------------------------------------")
#     print(sql_db_schema.invoke(table_name))

question = "Find the top 3 microbiological tests that were ordered the highest number of times for patients who had a percutaneous abdominal drainage (PAD) in the calendar year 2024, specifically considering only tests conducted within the same month as the PAD procedure. If there is more than one test with the same count at the third position, include all of them."
results = instruction_sql_search.invoke(question)
print(results)
