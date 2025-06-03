import os
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime
import logging
from typing import List, Dict, Any
from collections import defaultdict

# key utility functions
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

def save_results(results, filename: str):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {filename}")

def save_experiment_summary(summaries: List[Dict], filename: str = None):
    # Add timestamp to filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_summary_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(summaries, f, indent=2)
    logging.info(f"Experiment summaries saved to {filename}")

# task1 functions
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

# task1 analysis
def plot_sequence_length_vs_latency(results, filename: str = None):
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
    if not results["latencies"] or not results["request_indices"]:
        logging.warning("No results to plot")
        return
    
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
    if not results["throughputs"]:
        logging.warning("No results to plot")
        return
    
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
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task1_results/context_comparison_{timestamp}.png"
    
    plt.figure(figsize=(10, 6)) 
    context_ids = list(results_by_context.keys())
    values = [results_by_context[ctx_id][f"average_{metric}"] for ctx_id in context_ids]
    
    plt.bar(context_ids, values)
    plt.xlabel("Context ID")
    plt.ylabel(f"Average {metric.capitalize()}" + (" (seconds)" if metric == "latency" else " (requests/second)"))
    plt.title(f"Comparison of {metric.capitalize()} by Context")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    logging.info(f"Plot saved to {filename}")
    plt.close()

# task2 batch
def generate_batch_requests(contexts: Dict[str, str], questions: List[str], batch_size: int):
    # Generate all possible context+question combinations
    all_requests = []
    for ctx_id, ctx in contexts.items():
        for q in questions:
            all_requests.append({
                "context_id": ctx_id,
                "context": ctx,
                "question": q
            })
    random.shuffle(all_requests)
    # Generate batches
    for i in range(0, len(all_requests), batch_size):
        yield all_requests[i:i+batch_size]

def schedule_batch_requests(batch_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Schedule batch requests by grouping them by context_id.
    This reduces context switches and improves cache efficiency.
    """
    # Group requests by context_id
    context_groups = defaultdict(list)
    for request in batch_requests:
        context_id = request.get("context_id", -1)
        context_groups[context_id].append(request)
    
    # Flatten the grouped requests back into a list
    scheduled_requests = []
    for context_id, requests in context_groups.items():
        scheduled_requests.extend(requests)
    
    return scheduled_requests

def analyze_batch_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze batch processing results and calculate statistics.
    """
    if not results["latencies"]:
        return {"error": "No results to analyze"}
    
    analysis = {
        "total_requests": len(results["latencies"]),
        "average_latency": np.mean(results["latencies"]),
        "min_latency": np.min(results["latencies"]),
        "max_latency": np.max(results["latencies"]),
        "latency_std": np.std(results["latencies"]),
        "average_throughput": np.mean(results["throughputs"]) if results["throughputs"] else 0,
        "total_context_switches": sum(results["context_switches"]) if results["context_switches"] else 0,
        "average_context_switches": np.mean(results["context_switches"]) if results["context_switches"] else 0
    }
    
    return analysis

# task2 plotting 
def plot_batch_comparison(unscheduled_results: Dict[str, Any], scheduled_results: Dict[str, Any], 
                         filename: str = None):
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task2_results/batch_comparison_{timestamp}.png"
    
    # Extract batch metrics
    unscheduled_metrics = unscheduled_results.get("batch_metrics", [])
    scheduled_metrics = scheduled_results.get("batch_metrics", [])
    
    if not unscheduled_metrics or not scheduled_metrics:
        logging.warning("No batch metrics to plot")
        return
    
    # Extract data
    batch_ids = [m["batch_id"] for m in unscheduled_metrics]
    unscheduled_throughput = [m["throughput"] for m in unscheduled_metrics]
    scheduled_throughput = [m["throughput"] for m in scheduled_metrics]
    unscheduled_latency = [m["avg_latency"] for m in unscheduled_metrics]
    scheduled_latency = [m["avg_latency"] for m in scheduled_metrics]
    unscheduled_switches = [m["context_switches"] for m in unscheduled_metrics]
    scheduled_switches = [m["context_switches"] for m in scheduled_metrics]
    
    plt.figure(figsize=(15, 12))
    
    # Throughput comparison
    plt.subplot(3, 1, 1)
    bar_width = 0.35
    x = np.arange(len(batch_ids))
    plt.bar(x - bar_width/2, unscheduled_throughput, width=bar_width, 
            label="Unscheduled", color="blue", alpha=0.7)
    plt.bar(x + bar_width/2, scheduled_throughput, width=bar_width, 
            label="Scheduled", color="green", alpha=0.7)
    plt.xlabel("Batch ID")
    plt.ylabel("Throughput (requests/second)")
    plt.title("Throughput Comparison: Scheduled vs. Unscheduled")
    plt.xticks(x, batch_ids)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Latency comparison
    plt.subplot(3, 1, 2)
    plt.bar(x - bar_width/2, unscheduled_latency, width=bar_width, 
            label="Unscheduled", color="blue", alpha=0.7)
    plt.bar(x + bar_width/2, scheduled_latency, width=bar_width, 
            label="Scheduled", color="green", alpha=0.7)
    plt.xlabel("Batch ID")
    plt.ylabel("Average Latency (seconds)")
    plt.title("Latency Comparison: Scheduled vs. Unscheduled")
    plt.xticks(x, batch_ids)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Context switches comparison
    plt.subplot(3, 1, 3)
    plt.bar(x - bar_width/2, unscheduled_switches, width=bar_width, 
            label="Unscheduled", color="blue", alpha=0.7)
    plt.bar(x + bar_width/2, scheduled_switches, width=bar_width, 
            label="Scheduled", color="green", alpha=0.7)
    plt.xlabel("Batch ID")
    plt.ylabel("Context Switches")
    plt.title("Context Switches Comparison: Scheduled vs. Unscheduled")
    plt.xticks(x, batch_ids)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    logging.info(f"Batch comparison plot saved to {filename}")
    plt.close()

def plot_context_transitions(batch_results: Dict[str, Any], filename: str = None, 
                           title: str = "Context Transitions"):
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task2_results/context_transitions_{timestamp}.png"
    
    if not batch_results.get("context_ids"):
        logging.warning("No context IDs to plot")
        return
    
    context_ids = batch_results["context_ids"]
    request_indices = list(range(len(context_ids)))
    
    plt.figure(figsize=(12, 6))
    plt.plot(request_indices, context_ids, marker='o', linestyle='-')
    plt.xlabel("Request Sequence")
    plt.ylabel("Context ID")
    plt.title(title)
    plt.grid(True)
    
    # Count context switches
    context_switches = 0
    for i in range(1, len(context_ids)):
        if context_ids[i] != context_ids[i-1]:
            context_switches += 1
    
    plt.figtext(0.5, 0.01, f"Total context switches: {context_switches}", ha="center")
    
    plt.savefig(filename)
    logging.info(f"Context transitions plot saved to {filename}")
    plt.close()

def plot_batch_size_impact(batch_size_results: Dict[int, Dict], filename: str = None):
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task2_results/batch_size_impact_{timestamp}.png"
    
    batch_sizes = sorted(batch_size_results.keys())
    throughputs = [batch_size_results[size]["throughput"] for size in batch_sizes]
    latencies = [batch_size_results[size]["latency"] for size in batch_sizes]
    context_switches = [batch_size_results[size]["context_switches"] for size in batch_sizes]
    
    plt.figure(figsize=(15, 12))
    
    # Throughput plot
    plt.subplot(3, 1, 1)
    plt.plot(batch_sizes, throughputs, marker='o', linestyle='-', linewidth=2, color='blue')
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (requests/second)")
    plt.title("Impact of Batch Size on Throughput")
    plt.grid(True)
    
    # Add data labels
    for i, (size, throughput) in enumerate(zip(batch_sizes, throughputs)):
        plt.text(size, throughput + 0.05, f"{throughput:.2f}", ha='center')
    
    # Latency plot
    plt.subplot(3, 1, 2)
    plt.plot(batch_sizes, latencies, marker='o', linestyle='-', linewidth=2, color='orange')
    plt.xlabel("Batch Size")
    plt.ylabel("Average Latency (seconds)")
    plt.title("Impact of Batch Size on Latency")
    plt.grid(True)
    
    # Add data labels
    for i, (size, latency) in enumerate(zip(batch_sizes, latencies)):
        plt.text(size, latency + 0.05, f"{latency:.2f}", ha='center')
    
    # Context switches plot
    plt.subplot(3, 1, 3)
    plt.plot(batch_sizes, context_switches, marker='o', linestyle='-', linewidth=2, color='green')
    plt.xlabel("Batch Size")
    plt.ylabel("Context Switches")
    plt.title("Impact of Batch Size on Context Switches")
    plt.grid(True)
    
    # Add data labels
    for i, (size, switches) in enumerate(zip(batch_sizes, context_switches)):
        plt.text(size, switches + 0.5, f"{switches:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(filename)
    logging.info(f"Batch size impact plot saved to {filename}")
    plt.close()
