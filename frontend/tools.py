import os
import random
from typing import List, Dict

def read_chunks(file_folder) -> Dict[str, str]:
    """
    Read all the txt files in the folder and return the filenames and value pairs
    """
    filenames = os.listdir(file_folder)
    ret = {}
    for filename in filenames:
        if not filename.endswith("txt"):
            continue
        key = filename.removesuffix(".txt")
        with open(os.path.join(file_folder, filename), "r") as fin:
            value = fin.read()
        ret[key] = value
    return ret

def generate_requests(contexts: Dict[str, str], questions: List[str]):
    requests = []
    for context_id, context_text in contexts.items():
        for question in questions:
            requests.append({
                "context_id": context_id,
                "context": context_text,
                "question": question
            })
    for req in requests:
        yield req

def generate_requests_shuffle(contexts: Dict[str, str], questions: List[str]):
    requests = []
    for context_id, context_text in contexts.items():
        for question in questions:
            requests.append({
                "context_id": context_id,
                "context": context_text,
                "question": question
            })
    random.shuffle(requests)
    for req in requests:
        yield req