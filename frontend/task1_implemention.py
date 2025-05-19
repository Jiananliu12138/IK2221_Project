import time
import pandas as pd
import chat_session
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from tools import read_chunks, generate_requests, generate_requests_shuffle

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
chunks = read_chunks("./data_12/")

session = chat_session.ChatSession(IP1, PORT1)

system_prompt = (
    "You are a helpful assistant. I will now give you a document and "
    "please answer my question afterwards based on the content in document."
)
#task1 experiment1 分析 context+question 长度对延迟的影响
questions = [
    "What is the Abstract of this document?",
    "What is the Introduction of this document?",
    "What is the Conclusion of this document?",
    "What is the Background of this document?",
    "What is the KV Cache Manager?",
]

selected_keys = ["vllm_short", "vllm_medium", "vllm"]
task1_chunk = {k: v for k, v in chunks.items() if k in selected_keys}
#重复两遍看kv cache的效果
for repeat in range(2):
    results = []
    for req in generate_requests(task1_chunk, questions):
        session.set_context([system_prompt, req["context"]])
        start = time.perf_counter()
        try:
            response = "".join([x for x in session.chat(req["question"])])
        except Exception as e:
            print(f"Error: {e}")
            continue
        end = time.perf_counter()
        latency = end - start
        result = {
            "context_id": req["context_id"],
            "question": req["question"],
            "response": response,
            "latency": latency,
            "context_length": len(req["context"]),
            "question_length": len(req["question"]),
            "total_length": len(req["context"]) + len(req["question"])
        }
        results.append(result)
        print("="*40)
        print(f"Repeat: {repeat}")
        print(f"Context ID: {result['context_id']}")
        print(f"Question: {result['question']}")
        print(f"Response: {result['response'][:20]}...")
        print(f"Latency: {result['latency']:.3f} s")
        print(f"Context Length: {result['context_length']}")
        print(f"Question Length: {result['question_length']}")
        print(f"Total Length: {result['total_length']}")
        print(f"Throughput: {1/result['latency']:.2f} req/s")
        print("="*40)

    df = pd.DataFrame(results)
    df.to_csv(f"task1_result/task1_1_{repeat} results.csv", index=False)

    df_short = df[df["context_id"] == "vllm_short"]
    df_medium = df[df["context_id"] == "vllm_medium"]
    df_long = df[df["context_id"] == "vllm"]
    grouped_short = df_short.groupby("context_id")[["latency", "total_length"]].mean().reset_index()
    grouped_medium = df_medium.groupby("context_id")[["latency", "total_length"]].mean().reset_index()
    grouped_long = df_long.groupby("context_id")[["latency", "total_length"]].mean().reset_index()

    plt.figure(figsize=(8, 5))
    plt.plot(grouped_short["total_length"], grouped_short["latency"], marker='o', label="vllm_short")
    plt.plot(grouped_medium["total_length"], grouped_medium["latency"], marker='o', label="vllm_medium")
    plt.plot(grouped_long["total_length"], grouped_long["latency"], marker='o', label="vllm_long")
    plt.xlabel("Total Length")
    plt.ylabel("Average Latency (s)")
    plt.title("Latency vs Total Length for Different Contexts Lengths")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"task1_result/task1_1_{repeat}_plot.png")

#shuffle不能一块跑，token好像累计算的会过大
# for repeat in range(2,4):
#     results = []
#     for req in generate_requests_shuffle(task1_chunk, questions):
#         session.set_context([system_prompt, req["context"]])
#         start = time.perf_counter()
#         try:
#             response = "".join([x for x in session.chat(req["question"])])
#         except Exception as e:
#             print(f"Error: {e}")
#             continue
#         end = time.perf_counter()
#         latency = end - start
#         result = {
#             "context_id": req["context_id"],
#             "question": req["question"],
#             "response": response,
#             "latency": latency,
#             "context_length": len(req["context"]),
#             "question_length": len(req["question"]),
#             "total_length": len(req["context"]) + len(req["question"])
#         }
#         results.append(result)
#         print("="*40)
#         print(f"Repeat: {repeat}")
#         print(f"Context ID: {result['context_id']}")
#         print(f"Question: {result['question']}")
#         print(f"Response: {result['response'][:20]}...")
#         print(f"Latency: {result['latency']:.3f} s")
#         print(f"Context Length: {result['context_length']}")
#         print(f"Question Length: {result['question_length']}")
#         print(f"Total Length: {result['total_length']}")
#         print(f"Throughput: {1/result['latency']:.2f} req/s")
#         print("="*40)

#     df = pd.DataFrame(results)
#     df.to_csv(f"task1_result/task1_1_{repeat} results.csv", index=False)

#     df_short = df[df["context_id"] == "vllm_short"]
#     df_medium = df[df["context_id"] == "vllm_medium"]
#     df_long = df[df["context_id"] == "vllm"]
#     grouped_short = df_short.groupby("context_id")[["latency", "total_length"]].mean().reset_index()
#     grouped_medium = df_medium.groupby("context_id")[["latency", "total_length"]].mean().reset_index()
#     grouped_long = df_long.groupby("context_id")[["latency", "total_length"]].mean().reset_index()

#     plt.figure(figsize=(8, 5))
#     plt.plot(grouped_short["total_length"], grouped_short["latency"], marker='o', label="vllm_short")
#     plt.plot(grouped_medium["total_length"], grouped_medium["latency"], marker='o', label="vllm_medium")
#     plt.plot(grouped_long["total_length"], grouped_long["latency"], marker='o', label="vllm_long")
#     plt.xlabel("Total Length")
#     plt.ylabel("Average Latency (s)")
#     plt.title("Latency vs Total Length for Different Contexts Lengths")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#     plt.savefig(f"task1_result/task1_1_{repeat}_plot.png")

