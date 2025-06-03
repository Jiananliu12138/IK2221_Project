import os
import time
import numpy as np
import logging
import datetime
from typing import Dict, Any, List, Tuple
from chat_session import ChatSession
from tools import (
    read_chunks, save_results, save_experiment_summary,
    plot_batch_comparison, plot_context_transitions, plot_batch_size_impact,
    generate_batch_requests, schedule_batch_requests
)

IP = "192.168.2.27"
PORT = 8000

# Set up logging
log_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"task2_experiment_{log_timestamp}.log"
print(log_filename)
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Task2BatchRequestGenerator:
    def __init__(self, ip: str = IP, port: int = PORT):
        self.ip = ip
        self.port = port
        logger.info(f"Initializing with IP: {self.ip}, Port: {self.port}")
        
        # Initialize chat session
        self.chat_session = None
        self.contexts = {}  # Will store context_id -> content
        self.results = {
            "latencies": [],
            "sequence_lengths": [],
            "context_ids": [],
            "throughputs": [],
            "batch_sizes": [],
            "context_switches": [],
            "request_indices": []
        }
        self.initialize_chat_session()
        
    def initialize_chat_session(self):
        try:
            self.chat_session = ChatSession(ip=self.ip, port=self.port)
            logger.info("Chat session initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize chat session: {e}")
            return False
    
    def load_contexts(self, contexts_dir: str) -> None:
        logger.info(f"Loading contexts from directory: {contexts_dir}")
        self.contexts = read_chunks(contexts_dir)
        logger.info(f"Successfully loaded {len(self.contexts)} contexts from {contexts_dir}")
    
    def send_batch_requests(self, batch_requests: List[Dict[str, Any]], scheduled: bool = True) -> Tuple[List[float], List[int], float, int]:
        logger.info(f"Processing batch of {len(batch_requests)} requests (scheduled: {scheduled})")
        
        # Ensure we have a valid chat session
        if self.chat_session is None:
            if not self.initialize_chat_session():
                logger.error("Failed to initialize chat session")
                return [], [], -1, 0
        
        # scheduling if requested
        if scheduled:
            requests_to_process = schedule_batch_requests(batch_requests) # scheduler
        else:
            requests_to_process = batch_requests
        
        context_switches = 0 # count context switches to measure scheduling work
        prev_context = None
        for request in requests_to_process:
            context_id = request.get("context_id", -1)
            if prev_context is not None and context_id != prev_context:
                context_switches += 1
            prev_context = context_id
        logger.info(f"Context switches in batch: {context_switches}")
        
        # Process requests
        start_time = time.time()
        latencies = []
        sequence_lengths = []
        
        system_prompt = "You are a helpful assistant. I will now give you a document with one or more questions regarding it. Please answer my questions."
        
        prev_context_id = request["context_id"]
        for i, request in enumerate(requests_to_process): #the reqeust in batch is still sent one by one
            context_id = request["context_id"]
            context = request["context"]
            question = request["question"]
            
            # Calculate sequence length
            sequence_length = len(context) + len(question)
            sequence_lengths.append(sequence_length)
            logger.info(f"Processing request {i+1}/{len(requests_to_process)} for context {context_id}")
            
            try:
                if(i == 0):
                    self.chat_session.set_context([system_prompt, context])
                elif(prev_context_id != context_id):
                    self.chat_session.set_context([system_prompt, context])
                    prev_context_id = context_id
                
                request_start = time.perf_counter()
                response_text = ""
                for chunk in self.chat_session.chat(question): # send request
                    response_text += chunk
                
                latency = time.perf_counter() - request_start
                latencies.append(latency)
                logger.info(f"Request completed in {latency:.2f}s, response length: {len(response_text)} chars")
                
            except Exception as e:
                logger.error(f"Request failed: {e}")
                latencies.append(-1)
                # Try to reinitialize session
                self.initialize_chat_session()  
        total_time = time.time() - start_time
        
        # Filter out failed requests
        valid_latencies = [lat for lat in latencies if lat > 0]
        valid_sequence_lengths = [seq for i, seq in enumerate(sequence_lengths) if latencies[i] > 0]
        
        logger.info(f"Batch completed: {len(valid_latencies)}/{len(batch_requests)} successful requests in {total_time:.2f}s")
        
        return valid_latencies, valid_sequence_lengths, total_time, context_switches
    
    def run_batch_experiment(self, num_batches: int, batch_size: int, scheduled: bool = True, 
                            max_requests_per_batch: int = None) -> Dict[str, Any]:
        logger.info(f"\n===== Starting batch experiment =====")
        logger.info(f"Batches: {num_batches}, Batch size: {batch_size}, Scheduled: {scheduled}")
        self.results = {
            "latencies": [],
            "sequence_lengths": [],
            "context_ids": [],
            "throughputs": [],
            "batch_sizes": [],
            "context_switches": [],
            "request_indices": []
        }
        batch_metrics = []
        all_batch_results = []
        questions = [
            "What is the main topic of this document?",
            "Summarise the main ideas of the method in the document.",
            "What is the research context of this paper?",
            "How the main experiments in this paper were conducted?",
            "What exactly are the experimental results obtained from this question?",
            "What are the main conclusions in this document?"
        ]
        
        for batch_idx in range(num_batches):
            logger.info(f"\n----- Processing Batch {batch_idx+1}/{num_batches} -----")
            self.initialize_chat_session()
            
            all_batches = list(generate_batch_requests(self.contexts, questions, batch_size))
            batch_requests = all_batches[0] if all_batches else []
            
            # 限制批次大小，防止过载
            if max_requests_per_batch and len(batch_requests) > max_requests_per_batch:
                batch_requests = batch_requests[:max_requests_per_batch]
            
            # get batch results
            latencies, sequence_lengths, total_time, context_switches = self.send_batch_requests(
                batch_requests, scheduled=scheduled
            )
            
            successful_requests = len(latencies)
            throughput = successful_requests / total_time if total_time > 0 else 0
            avg_latency = np.mean(latencies) if latencies else 0

            batch_result = {
                "batch_id": batch_idx,
                "latencies": latencies,
                "sequence_lengths": sequence_lengths,
                "context_ids": [req["context_id"] for req in batch_requests[:successful_requests]],
                "throughput": throughput,
                "context_switches": context_switches,
                "successful_requests": successful_requests
            }
            all_batch_results.append(batch_result)
            
            self.results["latencies"].extend(latencies)
            self.results["sequence_lengths"].extend(sequence_lengths)
            self.results["context_ids"].extend(batch_result["context_ids"])
            self.results["throughputs"].append(throughput)
            self.results["batch_sizes"].append(len(batch_requests))
            self.results["context_switches"].append(context_switches)
            
            # add request indices 因为在批次处理中，需要保持全局连续的编号
            start_idx = len(self.results["request_indices"])
            self.results["request_indices"].extend(range(start_idx, start_idx + successful_requests))
            
            batch_metrics.append({
                "batch_id": batch_idx,
                "throughput": throughput,
                "avg_latency": avg_latency,
                "successful_requests": successful_requests,
                "context_switches": context_switches
            })
            
            logger.info(f"Batch {batch_idx+1} completed: {successful_requests} requests, {throughput:.2f} req/s, {context_switches} context switches")

        overall_throughput = np.mean(self.results["throughputs"]) if self.results["throughputs"] else 0
        overall_latency = np.mean(self.results["latencies"]) if self.results["latencies"] else 0
        overall_context_switches = np.mean(self.results["context_switches"]) if self.results["context_switches"] else 0
        
        logger.info(f"\n===== Experiment complete =====")
        logger.info(f"Total successful requests: {len(self.results['latencies'])}")
        logger.info(f"Average throughput: {overall_throughput:.2f} requests/s")
        logger.info(f"Average latency: {overall_latency:.2f} s")
        logger.info(f"Average context switches per batch: {overall_context_switches:.2f}")
        
        return {
            "overall_metrics": {
                "throughput": overall_throughput,
                "latency": overall_latency,
                "context_switches": overall_context_switches,
                "total_requests": len(self.results["latencies"])
            },
            "batch_metrics": batch_metrics,
            "raw_results": self.results,
            "all_batch_results": all_batch_results
        }
    
    def compare_scheduling_strategies(self, num_batches: int, batch_size: int, 
                                     max_requests_per_batch: int = None) -> Dict[str, Any]:
        logger.info(f"\n===== Comparing scheduling strategies =====")
        logger.info("\n----- Running UNSCHEDULED batch -----")
        unscheduled_results = self.run_batch_experiment(
            num_batches, batch_size, scheduled=False, 
            max_requests_per_batch=max_requests_per_batch
        )
        logger.info("\n----- Running SCHEDULED batch -----")
        scheduled_results = self.run_batch_experiment(
            num_batches, batch_size, scheduled=True, 
            max_requests_per_batch=max_requests_per_batch
        )

        logger.info(f"\n===== Scheduling Strategy Comparison =====")
        
        return {
            "unscheduled": unscheduled_results,
            "scheduled": scheduled_results
        }
    
    def vary_batch_size_experiment(self, batch_sizes: List[int], num_batches: int = 2, 
                                  scheduled: bool = True) -> Dict[str, Any]:
        logger.info(f"\n===== Analyzing impact of batch size =====")
        batch_size_results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"\n----- Testing batch size: {batch_size} -----")
            
            results = self.run_batch_experiment(
                num_batches, batch_size, scheduled=scheduled, 
                max_requests_per_batch=15 
            )
            
            batch_size_results[batch_size] = {
                "throughput": results["overall_metrics"]["throughput"],
                "latency": results["overall_metrics"]["latency"],
                "context_switches": results["overall_metrics"]["context_switches"],
                "raw_results": results
            }
        
        # Log results summary
        logger.info("\n===== Batch Size Impact Analysis =====")
        logger.info("Batch Size | Throughput (req/s) | Avg Latency (s) | Context Switches")
        logger.info("-----------|--------------------|-----------------|----------------")
        for size, result in batch_size_results.items():
            logger.info(f"{size:10d} | {result['throughput']:17.2f} | {result['latency']:15.2f} | {result['context_switches']:15.2f}")
        
        return batch_size_results

def main():
    # Get the current cache size from configuration.yaml
    try:
        import yaml
        with open("../../configuration.yaml", 'r') as f:
            config = yaml.safe_load(f)
            current_cache_size = config.get('max_local_cache_size', 'unknown')
    except Exception as e:
        logger.error(f"Error reading configuration.yaml: {e}")
        current_cache_size = "current"
    
    logger.info(f"\n===== Running Task 2 experiments with cache size: {current_cache_size} =====")

    os.makedirs("task2_results", exist_ok=True)
    cache_results_dir = f"task2_results/{current_cache_size}"
    os.makedirs(cache_results_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {cache_results_dir}")
    
    generator = Task2BatchRequestGenerator(ip=IP, port=PORT)
    generator.load_contexts("../data")
    
    experiment_summaries = []
    
    # [Exp1: Compare scheduling strategies with moderate context diversity]
    logger.info("\n===== Experiment 1: Scheduling Strategy Comparison =====")
    comparison_results = generator.compare_scheduling_strategies(
        num_batches=2,
        batch_size=10,
        max_requests_per_batch=15
    )
    
    #results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(comparison_results, f"{cache_results_dir}/exp1_scheduling_comparison_{timestamp}.json") 
    if (comparison_results["unscheduled"]["all_batch_results"] and 
        comparison_results["scheduled"]["all_batch_results"]):
        plot_batch_comparison(
            comparison_results["unscheduled"], 
            comparison_results["scheduled"],
            f"{cache_results_dir}/exp1_scheduling_comparison_{timestamp}.png"
        )
        plot_context_transitions(
            comparison_results["unscheduled"]["all_batch_results"][0],
            f"{cache_results_dir}/exp1_unscheduled_transitions_{timestamp}.png",
            "Unscheduled Context Transitions"
        )
        plot_context_transitions(
            comparison_results["scheduled"]["all_batch_results"][0],
            f"{cache_results_dir}/exp1_scheduled_transitions_{timestamp}.png",
            "Scheduled Context Transitions"
        )
    exp1_summary = {
        "experiment_name": "exp1_scheduling_comparison"
    }
    experiment_summaries.append(exp1_summary)
    
    # [Exp2: Impact of batch size]
    logger.info("\n===== Experiment 2: Impact of Batch Size =====")
    
    batch_size_results = generator.vary_batch_size_experiment(
        batch_sizes=[5, 10, 15],
        num_batches=2, 
        scheduled=True #with scheduling enabled for this one
    )
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(batch_size_results, f"{cache_results_dir}/exp2_batch_size_impact_{timestamp}.json")
    
    plot_batch_size_impact(
        batch_size_results,
        f"{cache_results_dir}/exp2_batch_size_impact_{timestamp}.png"
    )
    
    exp2_summary = {
        "experiment_name": "exp2_batch_size_impact",
        "batch_sizes_tested": list(batch_size_results.keys()),
        "best_batch_size": max(batch_size_results.keys(), 
                              key=lambda x: batch_size_results[x]["throughput"]),
        "throughput_range": [min(r["throughput"] for r in batch_size_results.values()),
                           max(r["throughput"] for r in batch_size_results.values())]
    }
    experiment_summaries.append(exp2_summary)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_experiment_summary(experiment_summaries, f"{cache_results_dir}/experiment_summary_{timestamp}.json")
    
    logger.info(f"\n===== All Task 2 experiments completed =====")
    logger.info(f"Current cache size: {current_cache_size}")
    logger.info(f"Log file: {log_filename}")
    logger.info(f"Results saved to: {cache_results_dir}")

if __name__ == "__main__":
    main()
