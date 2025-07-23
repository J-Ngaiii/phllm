import numpy as np
from pathlib import Path
import time
import pytest

# ---------- Helper functions to test ----------

def load_fna_files(directory):
    files = list(Path(directory).glob("*.fna"))
    sequences = []
    for f in files:
        with open(f, 'r') as file:
            seq = file.read()
            sequences.append(seq)
    return sequences

def save_embedding(embedding, bacteria_name, strain_or_phage, id, save_dir):
    timestamp = int(time.time())
    file_name = f"{bacteria_name}_{strain_or_phage}_{id}_{timestamp}.npy"
    full_path = save_dir / file_name
    np.save(full_path, embedding)
    return full_path

# Mock model for embedding test
class MockModel:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    def embed(self, input_list):
        # Return random embedding shaped (len(input_list), embedding_dim)
        return np.random.rand(len(input_list), self.embedding_dim)

# ---------- Tests ----------

def test_load_fna_files(tmp_path):
    # Create dummy .fna files in tmp directory
    file1 = tmp_path / "seq1.fna"
    file2 = tmp_path / "seq2.fna"
    file1.write_text(">seq1\nATCGATCG")
    file2.write_text(">seq2\nGGGCCC")

    sequences = load_fna_files(tmp_path)
    assert len(sequences) == 2
    assert all(isinstance(s, str) for s in sequences)
    assert ">seq1" in sequences[0]
    assert ">seq2" in sequences[1]

def test_embedding_model_output():
    model = MockModel(embedding_dim=128)
    sample_input = ["ATCG", "GGGCCC"]
    embeddings = model.embed(sample_input)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(sample_input), model.embedding_dim)
    assert np.all(np.isfinite(embeddings))

def test_save_and_load_embedding(tmp_path):
    embedding = np.random.rand(10)
    bacteria_name = "Ecoli"
    strain_or_phage = "strainX"
    id = "001"

    file_path = save_embedding(embedding, bacteria_name, strain_or_phage, id, tmp_path)

    loaded = np.load(file_path)
    assert np.array_equal(embedding, loaded)
    # Check filename format
    parts = file_path.name.split("_")
    assert parts[0] == bacteria_name
    assert parts[1] == strain_or_phage
    assert parts[2] == id
    assert parts[3].endswith(".npy") or parts[3].isdigit()  # timestamp + extension check

