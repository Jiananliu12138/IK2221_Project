import os
import random
from typing import List, Dict
from collections import defaultdict
#task1
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

#task2
def generate_batches(contexts, questions, batch_size):
    requests = []
    for ctx_id, ctx in contexts.items():
        for q in questions:
            requests.append({
                "context_id": ctx_id,
                "context": ctx,
                "question": q
            })
    # 按 batch_size 分组
    random.shuffle(requests)
    for i in range(0, len(requests), batch_size):
        yield requests[i:i+batch_size]

def scheduler(batch):
    groups = defaultdict(list)
    for req in batch:
        groups[req["context_id"]].append(req)
    ordered = []
    for group in groups.values():
        ordered.extend(group)
    return ordered