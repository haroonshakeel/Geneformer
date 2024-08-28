# ruff: noqa: F401
from .. import TOKEN_DICTIONARY_FILE
import pickle

def load_token_dictionary():
    with open(TOKEN_DICTIONARY_FILE, 'rb') as f:
        return pickle.load(f)

TOKEN_DICTIONARY = load_token_dictionary()

__all__ = ["TOKEN_DICTIONARY"]