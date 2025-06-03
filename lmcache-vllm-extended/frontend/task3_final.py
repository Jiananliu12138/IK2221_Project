import time
import pandas as pd
import chat_session
import matplotlib.pyplot as plt
import rag_utils
import logging
import datetime
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tools import read_chunks

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
IP1 = "192.168.2.27"
PORT1 = 8000

log_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./task3_result/task3_experiment_{log_timestamp}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
#所有文章键值对
chunks = read_chunks("./data/")

logger.info(f"Start to build RAG DB")

session = chat_session.ChatSession(IP1, PORT1)

system_prompt = (
    "You are a helpful assistant. I will now give you a document and "
    "please answer my question afterwards based on the content in document."
)

questions = {
    "What is the Abstract of CacheBlend?": "cacheblend",
    "What is the Introduction of CacheBlend?": "cacheblend",
    "What is the Conclusion of CacheBlend?": "cacheblend",
    "What is the Abstract of NFV Service Chains at the True Speed of the Underlying Hardware?": "metron-nsdi18",
    "What is the Introduction of NFV Service Chains at the True Speed of the Underlying Hardware?": "metron-nsdi18",
    "What is the Conclusion of NFV Service Chains at the True Speed of the Underlying Hardware?": "metron-nsdi18",
    "What is the Abstract of Efficient Memory Management for Large Language Model Serving with PagedAttention?": "vllm",
    "What is the Introduction of Efficient Memory Management for Large Language Model Serving with PagedAttention?": "vllm",
    "What is the Conclusion of Efficient Memory Management for Large Language Model Serving with PagedAttention?": "vllm",
}
plot_data = []
for i in range(3, 15, 4):
    rag_db_size = i
    rag_db = rag_utils.build_rag_db(chunks, model, tokenizer,rag_db_size)
    requests = []
    results = []
    for question, context_id in questions.items():
        if context_id in chunks:
            requests.append({
                "context_id": context_id,
                "context": chunks[context_id],
                "question": question
            })
    # 1. 生成请求（每个请求知道它的真实context_id，便于后续准确率评估）
    for req in requests:
    # 2. 检索最相关文档（RAG检索）
        t0 = time.perf_counter()
        retrieved_doc, sim = rag_utils.retrieve_context(req["question"], rag_db, model, tokenizer)
        t1 = time.perf_counter()
        retrieval_time = t1 - t0

        # 3. 用检索到的文档作为context，送入LLM推理
        session.set_context([system_prompt, retrieved_doc["text"]])
        t2 = time.perf_counter()
        response = "".join([x for x in session.chat(req["question"])])
        t3 = time.perf_counter()
        llm_time = t3 - t2
        total_time = t3 - t0

        # 4. 记录结果
        results.append({
            "expected_context_id": req["context_id"],
            "retrieved_context_id": retrieved_doc["doc_id"],
            "question": req["question"],
            "similarity": sim,
            "retrieval_time": retrieval_time,
            "llm_time": llm_time,
            "total_time": total_time,
            "hit": req["context_id"] == retrieved_doc["doc_id"],
            "response": response,
        })
        logger.info(f"rag_db大小{i}"+"="*40)
        logger.info(f"Q: {req['question']}")
        logger.info(f"Expected: {req['context_id']}, Retrieved: {retrieved_doc['doc_id']}, Hit: {req['context_id'] == retrieved_doc['doc_id']}, Similarity: {sim:.3f}")
        logger.info(f"Retrieval time: {retrieval_time:.2f}s, LLM time: {llm_time:.2f}s, Total: {total_time:.2f}s")
        logger.info(response[0:20] + "...")
        logger.info("="*40)

    df = pd.DataFrame(results)
    # 1. RAG 检索准确率
    accuracy = df["hit"].mean()
    logger.info(f"rag_db大小{i}"+"="*40)
    logger.info(f"RAG 检索准确率: {accuracy:.2%}")
    logger.info("平均检索时间: {:.3f}s".format(df["retrieval_time"].mean()))
    logger.info("平均LLM时间: {:.3f}s".format(df["llm_time"].mean()))
    logger.info("平均总延迟: {:.3f}s".format(df["total_time"].mean()))

    plot_data.append({
        "rag_db_size": rag_db_size,
        "accuracy": accuracy,
        "avg_retrieval_time": df["retrieval_time"].mean(),
        "avg_llm_time": df["llm_time"].mean(),
        "avg_total_time": df["total_time"].mean()
    })


rag_utils.save_results(plot_data, "./task3_result/task3_plot_data.json")
rag_utils.plot_results(plot_data, "./task3_result/task3_plot_acc.png","./task3_result/task3_plot_time.png")