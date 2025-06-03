import torch
import numpy as np
import logging
import json
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
#将文本输入的句子变为一维numpy数组
def get_llm_embedding(text, model, tokenizer, device="cpu"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1]
    embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

#生成rag_db数组
def build_rag_db(docs, model, tokenizer, chunk_size=3):
    logging.info(f"Building RAG DB, size = {chunk_size}...")
    required_keys = {"", "metron-nsdi18", "vllm"}  # 必须包含的键
    if chunk_size < len(required_keys):
        logging.error("Chunk size must be greater than or equal to the number of required keys.")
    # 确保必须包含的键存在于 docs 中
    required_docs = {key: docs[key] for key in required_keys if key in docs}
    # 将剩余的文档划分为块
    remaining_docs = {key: value for key, value in docs.items() if key not in required_keys}
    remaining_keys = list(remaining_docs.keys())
    chunk_docs = {}
    for i in range(chunk_size-len(required_keys)):
        chunk_docs[remaining_keys[i]] = remaining_docs[remaining_keys[i]]
        # 合并必须包含的文档
    chunk_docs.update(required_docs)
    # 构建 RAG 数据库
    chunk_rag_db = []
    for doc_id, text in chunk_docs.items():
        logging.info(f"Calculating document emb: {doc_id}")
        emb = get_llm_embedding(text, model, tokenizer)
        chunk_rag_db.append({"doc_id": doc_id, "text": text, "embedding": emb})
    logging.info("RAG DB built successfully.")
    return chunk_rag_db

#用于从 rag_db 中找出最相关的文档
def retrieve_context(question, rag_db, model, tokenizer):
    #将问题转成一维numpy数组
    q_emb = get_llm_embedding(question, model, tokenizer)
    doc_embs = np.stack([doc["embedding"] for doc in rag_db])
    sims = cosine_similarity([q_emb], doc_embs)[0]
    idx = np.argmax(sims)
    return rag_db[idx], sims[idx]

def save_results(results, filename: str):
    """
    Save results to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {filename}")

def plot_results(plot_data, filename_acc: str, filename_time: str):
    x = [data['rag_db_size'] for data in plot_data]
    accuracy = [data['accuracy'] for data in plot_data]
    plt.figure(figsize=(10, 6))
    plt.plot(x, accuracy, marker='o', label='Accuracy', color='blue')
    plt.title('RAG DB Size vs Accuracy')
    plt.xlabel('RAG DB Size')
    plt.ylabel('Accuracy')
    plt.legend()  
    plt.grid()
    plt.savefig(filename_acc)
    logging.info(f"Plot saved to {filename_acc}")
    plt.close()

    retrieval_time = [data['avg_retrieval_time'] for data in plot_data]
    llm_time = [data['avg_llm_time'] for data in plot_data]
    total_time = [data['avg_total_time'] for data in plot_data]
    plt.figure(figsize=(10, 6))
    plt.plot(x, retrieval_time, marker='o', label='Retrieval Time (s)', color='orange')
    plt.plot(x, llm_time, marker='o', label='LLM Time (s)', color='green')
    plt.plot(x, total_time, marker='o', label='Total Time (s)', color='red')
    plt.title('RAG DB Size vs Time')
    plt.xlabel('RAG DB Size')
    plt.ylabel('Time (s)')
    plt.legend()  
    plt.grid()
    plt.savefig(filename_time)
    logging.info(f"Plot saved to {filename_time}")
    plt.close()