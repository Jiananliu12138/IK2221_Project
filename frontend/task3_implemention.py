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
#所有文章键值对
chunks = read_chunks("./data/")

rag_db = rag_utils.build_rag_db(chunks, model, tokenizer)

session = chat_session.ChatSession(IP1, PORT1)

system_prompt = (
    "You are a helpful assistant. I will now give you a document and "
    "please answer my question afterwards based on the content in document."
)

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
  print("="*40)
  print(f"Q: {req['question']}")
  print(f"Expected: {req['context_id']}, Retrieved: {retrieved_doc['doc_id']}, Hit: {req['context_id'] == retrieved_doc['doc_id']}, Similarity: {sim:.3f}")
  print(f"Retrieval time: {retrieval_time:.2f}s, LLM time: {llm_time:.2f}s, Total: {total_time:.2f}s")
  print(response[0:20] + "...")
  print("="*40)

df = pd.DataFrame(results)
# 1. RAG 检索准确率
accuracy = df["hit"].mean()
print(f"RAG 检索准确率: {accuracy:.2%}")

# 2. 检索时间、LLM时间、总延迟的均值
print("平均检索时间: {:.3f}s".format(df["retrieval_time"].mean()))
print("平均LLM时间: {:.3f}s".format(df["llm_time"].mean()))
print("平均总延迟: {:.3f}s".format(df["total_time"].mean()))


plt.figure(figsize=(6,4))
plt.hist(df["retrieval_time"], bins=10, alpha=0.7, label="Retrieval")
plt.hist(df["llm_time"], bins=10, alpha=0.7, label="LLM")
plt.hist(df["total_time"], bins=10, alpha=0.7, label="Total")
plt.xlabel("Time (s)")
plt.ylabel("Count")
plt.title("Time Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("./task3_result/rag_time_hist.png")
plt.close()
