import time
import random
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import chat_session
import rag_utils
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel

# Change the following variables as needed
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
IP1 = "192.168.2.27"
PORT1 = 8000

@st.cache_resource
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


@st.cache_data
def read_chunks(file_folder) -> Dict[str, str]:
    """
    Read all the txt files in the folder and return the filenames
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
#键值对，文件名称对文章内容
chunks = read_chunks("./data/")
selected_chunks = st.multiselect(
    "Select the chunks into the context",
    list(chunks.keys()),
    default = [],
    placeholder = "Select in the drop-down menu")
contexts = [chunks[key] for key in selected_chunks]
rag_db = rag_utils.build_rag_db(chunks, model, tokenizer)


def generate_requests(contexts: Dict[str, str], questions: List[str]):
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


# task12
questions = [
    "What is the Abstract of this document?",
    "What is the Introduction of this document?",
    "What is the Conclusion of this document?",
]
if st.button("Run automated request experiment"):
    session = chat_session.ChatSession(IP1, PORT1)
    system_prompt = (
    "You are a helpful assistant. I will now give you a document and "
    "please answer my question afterwards based on the content in document."
    )
    all_contexts =  {k: chunks[k] for k in selected_chunks} if selected_chunks else chunks
    results = []
    #同步代码为啥老卡啊
    for req in generate_requests(all_contexts, questions):
        session.set_context([system_prompt, req["context"]])
        start = time.perf_counter()
        response = "".join([x for x in session.chat(req["question"])])
        end = time.perf_counter()
        latency = end - start
        results.append({
            "context_id": req["context_id"],
            "question": req["question"],
            "response": response,
            "latency": latency,
            "context_length": len(req["context"]),
            "question_length": len(req["question"]),
            "total_length": len(req["context"]) + len(req["question"])
        })
        st.write(f"Context: {req['context_id']}, Q: {req['question']}, Latency: {latency:.2f}s")
        st.write(response)
        st.divider()

    # 结果展示和保存
    df = pd.DataFrame(results)
    st.dataframe(df)
    st.line_chart(df, x="total_length", y="latency")
    df.to_csv("auto_request_results.csv", index=False)


#task3--先找到最相关的文档，然后用检索到的文档作为context，送入LLM推理
if st.button("Run RAG experiment"):
    results = []
    system_prompt_for_task3 = (
    "You are a helpful assistant. I will now give you a document and "
    "please answer my question afterwards based on the content in document."
    )
    mapping = {
    "What is the Abstract of The Click Modular Router?": "click",
    "What is the Introduction of NFV Service Chains at the True Speed of the Underlying Hardware?": "metron-nsdi18",
    "What is the Conclusion of Efficient Memory Management for Large Language Model Serving with PagedAttention?": "vllm",
}
    requests = []
    for question, context_id in mapping.items():
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
        session = chat_session.ChatSession(IP1,PORT1)
        session.set_context([system_prompt_for_task3, retrieved_doc["text"]])
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
        st.write(f"Q: {req['question']}")
        st.write(f"Expected: {req['context_id']}, Retrieved: {retrieved_doc['doc_id']}, Hit: {req['context_id'] == retrieved_doc['doc_id']}, Similarity: {sim:.3f}")
        st.write(f"Retrieval time: {retrieval_time:.2f}s, LLM time: {llm_time:.2f}s, Total: {total_time:.2f}s")
        st.write(response)
        st.divider()

        # 5. 结果统计与可视化
        # df = pd.DataFrame(results)
        # st.dataframe(df)
        # accuracy = df["hit"].mean() if len(df) > 0 else 0
        # st.write(f"RAG 检索准确率: {accuracy:.2%}")
        # st.line_chart(df, x="retrieval_time", y="total_time", use_container_width=True)
        # st.line_chart(df, x="retrieval_time", y="llm_time", use_container_width=True)
        # st.line_chart(df, x="retrieval_time", y="similarity", use_container_width=True)
        # df.to_csv("rag_experiment_results.csv", index=False)



#侧边栏
container = st.container(border=True)

with st.sidebar:
    system_prompt = st.sidebar.text_area(
        "System prompt:",
        "You are a helpful assistant. I will now give you a document and "
        "please answer my question afterwards based on the content in document",
    )

    session = chat_session.ChatSession(IP1,PORT1)
    session.set_context([system_prompt] + contexts)

    num_tokens = tokenizer.encode(session.get_context())
    container.header(f"The context given to LLM: ({len(num_tokens)} tokens)", divider = "grey")
    container.text(session.get_context())

    messages = st.container(height=300)
    messages.markdown("*vLLM instance 1*")
    if prompt := st.chat_input("Type the question here", key=1):
        messages.chat_message("user").write(prompt)
        messages.chat_message("assistant").write_stream(session.chat(prompt))
