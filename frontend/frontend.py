import time
import random
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import chat_session
from typing import List, Dict
from transformers import AutoTokenizer

# Change the following variables as needed
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
IP1 = "192.168.2.27"
PORT1 = 8000

@st.cache_resource
def get_tokenizer():
    global MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


tokenizer = get_tokenizer()


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

chunks = read_chunks("./data/")
selected_chunks = st.multiselect(
    "Select the chunks into the context",
    list(chunks.keys()),
    default = [],
    placeholder = "Select in the drop-down menu")
contexts = [chunks[key] for key in selected_chunks]

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
