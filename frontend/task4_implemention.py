import time
import pandas as pd
import chat_session
import matplotlib.pyplot as plt
import rag_utils
from transformers import AutoTokenizer, AutoModel
from tools import read_chunks, generate_batches, scheduler

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
IP1 = "192.168.2.27"
PORT1 = 8000

def get_tokenizer():
    global MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer

def get_model():
    global MODEL_NAME
    model = AutoModel.from_pretrained(MODEL_NAME)
    return model

tokenizer = get_tokenizer()

model = get_model().eval()

print(model.past_key_values)