# ruff: noqa: F401
import warnings
from pathlib import Path
import pickle

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")  # noqa # isort:skip


GENE_MEDIAN_FILE = Path(__file__).parent.parent / "gene_median_dictionary_gc95M.pkl"
# point to the actual location of the token dictionary
TOKEN_DICTIONARY_FILE = Path(__file__).parent.parent / "token_dictionary_gc95M.pkl"
ENSEMBL_DICTIONARY_FILE = Path(__file__).parent.parent / "gene_name_id_dict_gc95M.pkl"
ENSEMBL_MAPPING_FILE = Path(__file__).parent.parent / "ensembl_mapping_dict_gc95M.pkl"

# Load the token dictionary and other necessary files
with open(TOKEN_DICTIONARY_FILE, 'rb') as f:
    TOKEN_DICTIONARY = pickle.load(f)

with open(GENE_MEDIAN_FILE, 'rb') as f:
    GENE_MEDIAN_DICTIONARY = pickle.load(f)

with open(ENSEMBL_DICTIONARY_FILE, 'rb') as f:
    ENSEMBL_DICTIONARY = pickle.load(f)

with open(ENSEMBL_MAPPING_FILE, 'rb') as f:
    ENSEMBL_MAPPING = pickle.load(f)

# Make the loaded objects available to the classes in the mtl module
__all__ = [
    "TOKEN_DICTIONARY",
    "GENE_MEDIAN_DICTIONARY",
    "ENSEMBL_DICTIONARY",
    "ENSEMBL_MAPPING",
]
