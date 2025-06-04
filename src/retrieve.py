import json
import os

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_data(data_path, label_path, is_test=False):
    with open(os.path.join(data_path), "r") as f:
        data = json.load(f)
    if not is_test and label_path is not None:
        with open(os.path.join(label_path), "r") as f:
            labels = json.load(f)
    else:
        labels = {}
    return data, labels


class VectorDB:
    def __init__(
        self,
        dataset_path,
        index_path="./data/default.index",
        model_name="all-MiniLM-L6-v2",
    ):
        self.dataset_path = dataset_path
        self.index_path = index_path
        self.model_name = model_name
        if model_name == "emilyalsentzer/Bio_ClinicalBERT":
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.model = SentenceTransformer(model_name)
        self.df = pd.read_csv(dataset_path)
        self.data_dict = self.df.to_dict(orient="records")
        self.index = None

    def initialize(self):
        if not os.path.exists(self.index_path):
            self.build_index()
        else:
            self.load_index()

    def embed_text(self, query):
        if self.model_name == "all-MiniLM-L6-v2":
            return self.model.encode(query, convert_to_tensor=True).cpu()
        inputs = self.tokenizer(query, return_tensors="pt")
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embedding.detach().numpy()

    def save_index(self):
        faiss.write_index(self.index, str(self.index_path))

    def load_index(self):
        print(f"Loading index from {self.index_path}...")
        self.index = faiss.read_index(str(self.index_path))

    def build_index(self):
        embeddings = []

        print(f"Buidling index and save to {self.index_path} ...")
        # index all questions in the dataset and save the index
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            question = row["question"]
            embedding = self.embed_text(question)
            embeddings.append(embedding)

        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)  # type: ignore
        self.save_index()


class Retriever:
    def __init__(self, vector_db: VectorDB, top_n=3):
        self.vector_db = vector_db
        self.top_n = top_n

    def retrieve(self, question):
        """
        Given a user input, relevant splits are retrieved from storage using a Retriever.
        """

        query_embedding = np.array(
            [self.vector_db.embed_text(question)], dtype=np.float32
        )
        faiss.normalize_L2(query_embedding)
        if self.vector_db.index:
            distances, indices = self.vector_db.index.search(
                query_embedding, k=self.top_n
            )  # type: ignore
            results = [
                (self.vector_db.data_dict[idx], distances[0][i])
                for i, idx in enumerate(indices[0])
            ]
            return results
        return None


if __name__ == "__main__":
    # Load the dataset
    dataset_path = "./data/text_sql.csv"

    os.makedirs("./data/index", exist_ok=True)

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
    question = "Find the top 3 microbiological tests that were ordered the highest number of times for patients who had a percutaneous abdominal drainage (PAD) in the calendar year 2024, specifically considering only tests conducted within the same month as the PAD procedure. If there is more than one test with the same count at the third position, include all of them."
    results = retriever.retrieve(question)
    print(f"Query: {question}")
    if results:
        for result, distance in results:
            print(f"Question: {result['question']}, Distance: {distance:.4f}")
    else:
        print("No results found.")
