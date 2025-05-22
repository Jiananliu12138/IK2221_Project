import os
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime
import logging
from typing import List, Dict, Any
from collections import defaultdict

#task1 utility functions
def read_chunks(file_folder) -> Dict[str, str]:
    """
    Read all the txt files in the data folder and return the filenames as key and contents as value, into a dictionary and return it.
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


#task1 plotting and analysis functions
def plot_sequence_length_vs_latency(results, filename: str = None):
    """
    Plot the relationship between sequence length and latency.
    """
    if not results["latencies"] or not results["sequence_lengths"]:
        logging.warning("No results to plot")
        return
    
    # Add timestamp to filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task1_results/sequence_length_vs_latency_{timestamp}.png"
    
    plt.figure(figsize=(10, 6))
    plt.scatter(results["sequence_lengths"], results["latencies"])
    
    # Add trend line
    z = np.polyfit(results["sequence_lengths"], results["latencies"], 1)
    p = np.poly1d(z)
    plt.plot(results["sequence_lengths"], p(results["sequence_lengths"]), "r--", 
             label=f"Trend: y={z[0]:.6f}x+{z[1]:.6f}")
    
    plt.xlabel("Sequence Length (characters)")
    plt.ylabel("Latency (seconds)")
    plt.title("Effect of Sequence Length on Response Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    logging.info(f"Plot saved to {filename}")
    plt.close()

def plot_request_number_vs_latency(results, filename: str = None):
    """
    Plot how latency changes over sequential requests (useful for cache analysis).
    """
    if not results["latencies"] or not results["request_indices"]:
        logging.warning("No results to plot")
        return
    
    # Add timestamp to filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task1_results/request_number_vs_latency_{timestamp}.png"
    
    plt.figure(figsize=(10, 6))
    plt.plot(results["request_indices"], results["latencies"], marker='o')
    plt.xlabel("Request Number")
    plt.ylabel("Latency (seconds)")
    plt.title("Latency Over Sequential Requests")
    plt.grid(True)
    plt.savefig(filename)
    logging.info(f"Plot saved to {filename}")
    plt.close()

def plot_throughput_over_time(results, filename: str = None):
    """
    Plot throughput over time.
    """
    if not results["throughputs"]:
        logging.warning("No results to plot")
        return
    
    # Add timestamp to filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task1_results/throughput_over_time_{timestamp}.png"
    
    request_indices = list(range(1, len(results["throughputs"]) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(request_indices, results["throughputs"], marker='o')
    plt.xlabel("Request Number")
    plt.ylabel("Throughput (requests/second)")
    plt.title("Throughput Over Time")
    plt.grid(True)
    plt.savefig(filename)
    logging.info(f"Plot saved to {filename}")
    plt.close()

def analyze_results_by_context(results) -> Dict[str, Dict]:
    """
    Analyze results grouped by context.
    
    Returns:
        Dictionary of context_id -> analysis results
    """
    context_ids = set(results["context_ids"])
    analysis = {}
    
    for context_id in context_ids:
        # Get indices for this context
        indices = [i for i, ctx in enumerate(results["context_ids"]) if ctx == context_id]
        
        # Extract data for this context
        latencies = [results["latencies"][i] for i in indices]
        sequence_lengths = [results["sequence_lengths"][i] for i in indices]
        throughputs = [results["throughputs"][i] for i in indices]
        
        # Calculate statistics
        analysis[context_id] = {
            "count": len(indices),
            "average_latency": np.mean(latencies),
            "min_latency": np.min(latencies),
            "max_latency": np.max(latencies),
            "average_sequence_length": np.mean(sequence_lengths),
            "average_throughput": np.mean(throughputs)
        }
    
    return analysis

def plot_context_comparison(results_by_context: Dict[str, Dict], 
                           metric: str = "latency", 
                           filename: str = None):
    """
    Plot comparison of different contexts.
    """
    # Add timestamp to filename
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task1_results/context_comparison_{timestamp}.png"
    
    plt.figure(figsize=(10, 6))
    
    context_ids = list(results_by_context.keys())
    values = [results_by_context[ctx_id][f"average_{metric}"] for ctx_id in context_ids] #get the average latency or throughput key from the dict
    
    plt.bar(context_ids, values)
    plt.xlabel("Context ID")
    plt.ylabel(f"Average {metric.capitalize()}" + (" (seconds)" if metric == "latency" else " (requests/second)"))
    plt.title(f"Comparison of {metric.capitalize()} by Context")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    logging.info(f"Plot saved to {filename}")
    plt.close()

def save_results(results, filename: str):
    """
    Save results to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {filename}")

def save_experiment_summary(summaries: List[Dict], filename: str = None):
    """
    Save experiment summaries to a JSON file.
    """
    # Add timestamp to filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task1_results/experiment_summary_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(summaries, f, indent=2)
    logging.info(f"Experiment summaries saved to {filename}")


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
import os
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime
import logging
from typing import List, Dict, Any
from collections import defaultdict

#task1 utility functions
def read_chunks(file_folder) -> Dict[str, str]:
    """
    Read all the txt files in the data folder and return the filenames as key and contents as value, into a dictionary and return it.
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


#task1 plotting and analysis functions
def plot_sequence_length_vs_latency(results, filename: str = None):
    """
    Plot the relationship between sequence length and latency.
    """
    if not results["latencies"] or not results["sequence_lengths"]:
        logging.warning("No results to plot")
        return
    
    # Add timestamp to filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task1_results/sequence_length_vs_latency_{timestamp}.png"
    
    plt.figure(figsize=(10, 6))
    plt.scatter(results["sequence_lengths"], results["latencies"])
    
    # Add trend line
    z = np.polyfit(results["sequence_lengths"], results["latencies"], 1)
    p = np.poly1d(z)
    plt.plot(results["sequence_lengths"], p(results["sequence_lengths"]), "r--", 
             label=f"Trend: y={z[0]:.6f}x+{z[1]:.6f}")
    
    plt.xlabel("Sequence Length (characters)")
    plt.ylabel("Latency (seconds)")
    plt.title("Effect of Sequence Length on Response Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    logging.info(f"Plot saved to {filename}")
    plt.close()

def plot_request_number_vs_latency(results, filename: str = None):
    """
    Plot how latency changes over sequential requests (useful for cache analysis).
    """
    if not results["latencies"] or not results["request_indices"]:
        logging.warning("No results to plot")
        return
    
    # Add timestamp to filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task1_results/request_number_vs_latency_{timestamp}.png"
    
    plt.figure(figsize=(10, 6))
    plt.plot(results["request_indices"], results["latencies"], marker='o')
    plt.xlabel("Request Number")
    plt.ylabel("Latency (seconds)")
    plt.title("Latency Over Sequential Requests")
    plt.grid(True)
    plt.savefig(filename)
    logging.info(f"Plot saved to {filename}")
    plt.close()

def plot_throughput_over_time(results, filename: str = None):
    """
    Plot throughput over time.
    """
    if not results["throughputs"]:
        logging.warning("No results to plot")
        return
    
    # Add timestamp to filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task1_results/throughput_over_time_{timestamp}.png"
    
    request_indices = list(range(1, len(results["throughputs"]) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(request_indices, results["throughputs"], marker='o')
    plt.xlabel("Request Number")
    plt.ylabel("Throughput (requests/second)")
    plt.title("Throughput Over Time")
    plt.grid(True)
    plt.savefig(filename)
    logging.info(f"Plot saved to {filename}")
    plt.close()

def analyze_results_by_context(results) -> Dict[str, Dict]:
    """
    Analyze results grouped by context.
    
    Returns:
        Dictionary of context_id -> analysis results
    """
    context_ids = set(results["context_ids"])
    analysis = {}
    
    for context_id in context_ids:
        # Get indices for this context
        indices = [i for i, ctx in enumerate(results["context_ids"]) if ctx == context_id]
        
        # Extract data for this context
        latencies = [results["latencies"][i] for i in indices]
        sequence_lengths = [results["sequence_lengths"][i] for i in indices]
        throughputs = [results["throughputs"][i] for i in indices]
        
        # Calculate statistics
        analysis[context_id] = {
            "count": len(indices),
            "average_latency": np.mean(latencies),
            "min_latency": np.min(latencies),
            "max_latency": np.max(latencies),
            "average_sequence_length": np.mean(sequence_lengths),
            "average_throughput": np.mean(throughputs)
        }
    
    return analysis

def plot_context_comparison(results_by_context: Dict[str, Dict], 
                           metric: str = "latency", 
                           filename: str = None):
    """
    Plot comparison of different contexts.
    """
    # Add timestamp to filename
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task1_results/context_comparison_{timestamp}.png"
    
    plt.figure(figsize=(10, 6))
    
    context_ids = list(results_by_context.keys())
    values = [results_by_context[ctx_id][f"average_{metric}"] for ctx_id in context_ids] #get the average latency or throughput key from the dict
    
    plt.bar(context_ids, values)
    plt.xlabel("Context ID")
    plt.ylabel(f"Average {metric.capitalize()}" + (" (seconds)" if metric == "latency" else " (requests/second)"))
    plt.title(f"Comparison of {metric.capitalize()} by Context")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    logging.info(f"Plot saved to {filename}")
    plt.close()

def save_results(results, filename: str):
    """
    Save results to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {filename}")

def save_experiment_summary(summaries: List[Dict], filename: str = None):
    """
    Save experiment summaries to a JSON file.
    """
    # Add timestamp to filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task1_results/experiment_summary_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(summaries, f, indent=2)
    logging.info(f"Experiment summaries saved to {filename}")


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
