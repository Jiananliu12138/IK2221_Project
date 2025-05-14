import torch
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
def build_rag_db(docs, model, tokenizer):
    rag_db = []
    for doc_id, text in docs.items():
        emb = get_llm_embedding(text, model, tokenizer)
        rag_db.append({"doc_id": doc_id, "text": text, "embedding": emb})
    return rag_db
#用于从 rag_db 中找出最相关的文档
def retrieve_context(question, rag_db, model, tokenizer):
    #将问题转成一维numpy数组
    q_emb = get_llm_embedding(question, model, tokenizer)
    doc_embs = np.stack([doc["embedding"] for doc in rag_db])
    sims = cosine_similarity([q_emb], doc_embs)[0]
    idx = np.argmax(sims)
    return rag_db[idx], sims[idx]