import os
import time
import numpy as np
import logging
import datetime
from typing import Dict, Any, Tuple
from chat_session import ChatSession
from tools import (
    read_chunks, generate_requests, generate_requests_shuffle,
    plot_sequence_length_vs_latency, plot_request_number_vs_latency,
    plot_throughput_over_time, analyze_results_by_context,
    plot_context_comparison, save_results, save_experiment_summary
)

IP = "192.168.2.27"
PORT = 8000

# Set up logging
log_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"task1_experiment_{log_timestamp}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Task1RequestGenerator:
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
            "request_indices": []
        }
        
        # Initialize chat session
        self.initialize_chat_session()
        
    def initialize_chat_session(self):
        try:
            self.chat_session = ChatSession(ip=self.ip, port=self.port)
            logger.info("Chat session initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize chat session: {e}")
            return False
    
    def load_contexts(self,contexts_dir: str) -> None:
        logger.info(f"Loading contexts from directory: {contexts_dir}")
        self.contexts = read_chunks(contexts_dir) #dict[filename, content]
        logger.info(f"Successfully loaded {len(self.contexts)} contexts from {contexts_dir}")
    
    def send_request(self,request: Dict[str, Any]) -> Tuple[float, int]:
        context_id = request["context_id"]
        context = request["context"]
        question = request["question"]
        
        # Ensure we have a valid chat session
        if self.chat_session is None:
            if not self.initialize_chat_session():
                return -1, -1
        
        system_prompt = "You are a helpful assistant. I will now give you a document and please answer my question afterwards based on the content in document."
        
        try:
            self.chat_session.set_context([system_prompt, context]) #set final context to model
        except Exception as e:
            logger.error(f"Error setting context: {e}")
            if self.initialize_chat_session():
                try:
                    self.chat_session.set_context([system_prompt, context])
                except Exception:
                    return -1, -1
            else:
                return -1, -1
        
        # Calculate sequence length 
        sequence_length = len(context) + len(question)
        
        logger.info(f"Sending question for context {context_id}: '{question[:30]}...'")
        start_time = time.perf_counter()
        
        try:
            response_text = ""
            for chunk in self.chat_session.chat(question): #messages[final_context + question] for sending
                response_text += chunk
            
            latency = time.perf_counter() - start_time
            logger.info(f"Request completed in {latency:.2f}s, response length: {len(response_text)} chars")
            return latency, sequence_length
        except Exception as e:
            logger.error(f"Request failed: {e}")
            self.initialize_chat_session()
            return -1, -1
    
    def run_experiment_with_generator(self, request_generator, experiment_name: str, max_requests: int = None, results_dir: str = "task1_results") -> Dict[str, Any]:
        logger.info(f"\n===== Starting experiment: {experiment_name} =====")
        
        self.results = {
            "latencies": [],
            "sequence_lengths": [],
            "context_ids": [],
            "throughputs": [],
            "request_indices": []
        }
        
        start_time = time.time()
        request_count = 0
        
        for request in request_generator:
            request_count += 1
            if max_requests and request_count > max_requests:
                break
                
            context_id = request["context_id"]
            
            latency, sequence_length = self.send_request(request)
            
            if latency > 0:
                self.results["latencies"].append(latency)
                self.results["sequence_lengths"].append(sequence_length)
                self.results["context_ids"].append(context_id)
                self.results["request_indices"].append(request_count - 1)
                
                # Calculate current throughput (requests/s)
                current_time = time.time()
                elapsed = current_time - start_time
                throughput = len(self.results["latencies"]) / elapsed if elapsed > 0 else 0
                self.results["throughputs"].append(throughput)
                
                logger.info(f"Request {request_count} - Context: {context_id}, Length: {sequence_length}, Latency: {latency:.2f}s, Throughput: {throughput:.2f} req/s")
            else:
                logger.warning(f"Request {request_count} failed - Context: {context_id}")
        
        # Calculate overall throughput and average latency
        total_time = time.time() - start_time
        overall_throughput = len(self.results["latencies"]) / total_time if total_time > 0 else 0
        avg_latency = np.mean(self.results["latencies"]) if self.results["latencies"] else 0
        
        logger.info(f"\n===== Experiment complete: {experiment_name} =====")
        logger.info(f"Successful requests: {len(self.results['latencies'])}/{request_count}")
        logger.info(f"Overall throughput: {overall_throughput:.2f} requests/second")
        logger.info(f"Average latency: {avg_latency:.2f} seconds")
        
        save_results(self.results, f"{results_dir}/{experiment_name}_results.json")
        
        summary = {
            "experiment_name": experiment_name,
            "num_requests": request_count,
            "successful_requests": len(self.results["latencies"]),
            "overall_throughput": overall_throughput,
            "average_latency": avg_latency
        }
        return summary
    


def main():
    # Get the current cache size from configuration.yaml（we need change cache size manually for ）
    try:
        import yaml
        with open("../configuration.yaml", 'r') as f:
            config = yaml.safe_load(f)
            current_cache_size = config.get('max_local_cache_size', 'unknown')
    except Exception as e:
        logger.error(f"Error reading configuration.yaml: {e}")
        current_cache_size = "current"
    
    logger.info(f"\n===== Running Task 1 experiments with cache size: {current_cache_size} =====")
    
    # Create results directory with cache size subfolder
    os.makedirs("task1_results", exist_ok=True)
    cache_results_dir = f"task1_results/{current_cache_size}"
    os.makedirs(cache_results_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {cache_results_dir}")
    
    generator = Task1RequestGenerator(ip=IP, port=PORT)
    generator.load_contexts("./data")

    questions = [
        "What is the main topic of this document?",
        "Summarise the main ideas of the method in the document.",
        "What is the research context of this paper?",
        "How the main experiments in this paper were conducted?",
        "What exactly are the experimental results obtained from this question?",
        "What are the main conclusions in this document?"
    ]
    
    experiment_summaries = []
    
    # [Exp1: Measure latency vs sequence length with different context lengths]
    logger.info("\n===== Experiment 1: Latency vs Sequence Length =====")

    exp1_generator = generate_requests(generator.contexts, questions)
    exp1_summary = generator.run_experiment_with_generator(exp1_generator, "exp1_sequence_length", max_requests=10, results_dir=cache_results_dir)
    experiment_summaries.append(exp1_summary)
    
    # Plot results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_sequence_length_vs_latency(generator.results, f"{cache_results_dir}/Exp1_sequence_length_vs_latency_{timestamp}.png")
 
    context_analysis = analyze_results_by_context(generator.results) #Groups requests by context and performs statistical analyses on each group of requests
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_context_comparison(context_analysis, "latency", f"{cache_results_dir}/Exp1_context_comparison_{timestamp}.png")
    

    # [Exp2: Test cache reuse with repeated context]
    logger.info("\n===== Experiment 2: Cache Reuse with Repeated Context =====")
    
    # Select a single context for repeated use：Would the response be faster if I asked the same question (repeated twice) consecutively for the same document?

    if len(generator.contexts) > 0:
        context_id = list(generator.contexts.keys())[0] # take the first context as an example
        single_context = {context_id: generator.contexts[context_id]}
        
        repeated_requests = []
        for _ in range(2):  # Run the same questions twice
            repeated_requests.extend(generate_requests(single_context, questions))
        
        # Run the experiment
        exp2_summary = generator.run_experiment_with_generator(iter(repeated_requests),  "exp2_cache_reuse", results_dir=cache_results_dir) #iter 逐条拉取请求并发送(per question)
        experiment_summaries.append(exp2_summary)
        
        # Plot results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_request_number_vs_latency(generator.results, f"{cache_results_dir}/Exp2_request_number_vs_latency_{timestamp}.png")
    
    
    # [Exp3: Impact of Context Diversity]
    logger.info("\n===== Experiment 3: Impact of Context Diversity =====")
    
    # {Low diversity} - Using only 2 contexts for better cache reuse
    logger.info("Running with low diversity (only 2 contexts)...")
    if len(generator.contexts) >= 2:
        context_keys = list(generator.contexts.keys())
        low_diversity_contexts = {k: generator.contexts[k] for k in context_keys[:2]} #use only the first 2 contexts for low diversity
        
        low_diversity_generator = generate_requests(low_diversity_contexts, questions)
        
        exp3a_summary = generator.run_experiment_with_generator(low_diversity_generator, "exp3a_low_diversity", max_requests=15, results_dir=cache_results_dir)
        experiment_summaries.append(exp3a_summary)
        
        # Plot results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_request_number_vs_latency(generator.results, f"{cache_results_dir}/Exp3a_low_diversity_latency_{timestamp}.png")
        plot_throughput_over_time(generator.results, f"{cache_results_dir}/Exp3a_low_diversity_throughput_{timestamp}.png")
    
    # {High diversity} - Using all available contexts with shuffling for minimum cache reuse
    logger.info("Running with high diversity (all contexts shuffled)...")
    if len(generator.contexts) >= 3:
        high_diversity_contexts = generator.contexts #use all contexts for high diversity

        high_diversity_generator = generate_requests_shuffle(high_diversity_contexts, questions) #generate_requests_shuffle 生成请求并打乱顺序
        
        exp3b_summary = generator.run_experiment_with_generator(high_diversity_generator, "exp3b_high_diversity", max_requests=15, results_dir=cache_results_dir)
        experiment_summaries.append(exp3b_summary) #the requ
        
        # Plot results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_request_number_vs_latency(generator.results, f"{cache_results_dir}/Exp3b_high_diversity_latency_{timestamp}.png")
        plot_throughput_over_time(generator.results, f"{cache_results_dir}/Exp3b_high_diversity_throughput_{timestamp}.png")
    
    # Save all experiment summaries
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_experiment_summary(experiment_summaries, f"{cache_results_dir}/experiment_summary_{timestamp}.json")

    logger.info(f"Current cache size: {current_cache_size}")
    logger.info(f"Log file: {log_filename}")
    

if __name__ == "__main__":
    main()
