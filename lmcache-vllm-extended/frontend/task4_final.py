import time
import pandas as pd
import chat_session
import matplotlib.pyplot as plt
import rag_utils
from transformers import AutoTokenizer, AutoModel
from tools import read_chunks, generate_batches, scheduler
import torch
import kv_utils

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

chunks = read_chunks("./data/")

questions = {
    "What is the Abstract of The Click Modular Router?": "click",
    "What is the Introduction of The Click Modular Router?": "click",
    "What is the Conclusion of The Click Modular Router?": "click",
    "What is the Abstract of NFV Service Chains at the True Speed of the Underlying Hardware?": "metron-nsdi18",
    "What is the Introduction of NFV Service Chains at the True Speed of the Underlying Hardware?": "metron-nsdi18",
    "What is the Conclusion of NFV Service Chains at the True Speed of the Underlying Hardware?": "metron-nsdi18",
    "What is the Abstract of Efficient Memory Management for Large Language Model Serving with PagedAttention?": "vllm",
    "What is the Introduction of Efficient Memory Management for Large Language Model Serving with PagedAttention?": "vllm",
    "What is the Conclusion of Efficient Memory Management for Large Language Model Serving with PagedAttention?": "vllm",
}

requests = []

kv_list = []

labels = []

system_prompt = (
    "You are a helpful assistant. I will now give you a document and "
    "please answer my question afterwards based on the content in document."
)

for question, context_id in questions.items():
  if context_id in chunks:
    requests.append({
        "context_id": context_id,
        "context": chunks[context_id],
        "question": question
    })

for req in requests:
    # 构建输入（可拼接 system_prompt、context、question）
    # input_text = system_prompt + "\n" + req["context"] + "\n" + req["question"]
    input_text = system_prompt  + "\n" + req["question"]
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
    kv = outputs.past_key_values  # list: num_layers × (key, value)
    kv_list.append(kv)
    labels.append(req["context_id"])  # 真实主题标签
    print(f"请求ID: {req['context_id']}")
# 请求ID: click, KV张量大小: 28, 形状: 
# [torch.Size([1, 2, 709, 128]), torch.Size([1, 2, 709, 128]), torch.Size([1, 2, 709, 128]), 
#  torch.Size([1, 2, 709, 128]), torch.Size([1, 2, 709, 128]), torch.Size([1, 2, 709, 128]), 
#  torch.Size([1, 2, 709, 128]), torch.Size([1, 2, 709, 128]), torch.Size([1, 2, 709, 128]), 
#  torch.Size([1, 2, 709, 128]), torch.Size([1, 2, 709, 128]), torch.Size([1, 2, 709, 128]), 
#  torch.Size([1, 2, 709, 128]), torch.Size([1, 2, 709, 128]), torch.Size([1, 2, 709, 128]), 
#  torch.Size([1, 2, 709, 128]), torch.Size([1, 2, 709, 128]), torch.Size([1, 2, 709, 128]), 
#  torch.Size([1, 2, 709, 128]), torch.Size([1, 2, 709, 128]), torch.Size([1, 2, 709, 128]), 
#  torch.Size([1, 2, 709, 128]), torch.Size([1, 2, 709, 128]), torch.Size([1, 2, 709, 128]), 
#  torch.Size([1, 2, 709, 128]), torch.Size([1, 2, 709, 128]), torch.Size([1, 2, 709, 128]), 
#  torch.Size([1, 2, 709, 128])]

# kv[0] 是第0层的 (key, value) 二元组
# print(type(kv), len(kv))
# print(kv[0][0].shape)  # 第0层的 key 张量的shape
# <class 'transformers.cache_utils.DynamicCache'> 28
# torch.Size([1, 2, 1, 128])
#(batch_size, num_heads, seq_len, head_dim)

ari, nmi, cluster_labels = kv_utils.kv_clustering_pipeline(
    kv_list, labels, n_clusters=len(set(labels)), agg_method="mean"
)
print(f"KV聚类 ARI: {ari:.3f}, NMI: {nmi:.3f}")