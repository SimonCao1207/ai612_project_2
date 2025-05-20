from src.envs.mimic_iv.tools.sql_db_schema import SqlDbSchema
from sqlalchemy import create_engine
import json

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

for table_name in list_of_tables:
    print("-------------------------------------------")
    print(sql_db_schema.invoke(table_name))