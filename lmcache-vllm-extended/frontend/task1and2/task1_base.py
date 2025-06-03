import os
import time
import numpy as np
import logging
import datetime
from typing import Dict, Any, Tuple
from chat_session import ChatSession
from tools import (
    read_chunks,  save_results
)

IP = "192.168.2.27"
PORT = 8000

class Task1RequestGenerator:
    def __init__(self, ip: str = IP, port: int = PORT):
        self.ip = ip
        self.port = port
        
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
            logging.info("Chat session initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize chat session: {e}")
            return False
    
    def load_contexts(self, contexts_dir: str) -> None:
        logging.info(f"Loading contexts from directory: {contexts_dir}")
        self.contexts = read_chunks(contexts_dir) #dict[filename, content]
        logging.info(f"Successfully loaded {len(self.contexts)} contexts from {contexts_dir}")
    
    def send_request(self, request: Dict[str, Any]) -> Tuple[float, int]:
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
            logging.error(f"Error setting context: {e}")
            if self.initialize_chat_session():
                try:
                    self.chat_session.set_context([system_prompt, context])
                except Exception:
                    return -1, -1
            else:
                return -1, -1
        
        # Calculate sequence length 
        sequence_length = len(context) + len(question)
        
        logging.info(f"Sending question for context {context_id}: '{question[:30]}...'")
        start_time = time.perf_counter()
        
        try:
            response_text = ""
            for chunk in self.chat_session.chat(question): #messages[final_context + question] for sending
                response_text += chunk
            
            latency = time.perf_counter() - start_time
            logging.info(f"Request completed in {latency:.2f}s, response length: {len(response_text)} chars")
            return latency, sequence_length
        except Exception as e:
            logging.error(f"Request failed: {e}")
            self.initialize_chat_session()
            return -1, -1
    
    def run_experiment_with_generator(self, request_generator, experiment_name: str, max_requests: int = None, results_dir: str = "task1_results") -> Dict[str, Any]:
        logging.info(f"\n===== Starting experiment: {experiment_name} =====")
        
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
                
                logging.info(f"Request {request_count} - Context: {context_id}, Length: {sequence_length}, Latency: {latency:.2f}s, Throughput: {throughput:.2f} req/s")
            else:
                logging.warning(f"Request {request_count} failed - Context: {context_id}")
        
        # Calculate overall throughput and average latency
        total_time = time.time() - start_time
        overall_throughput = len(self.results["latencies"]) / total_time if total_time > 0 else 0
        avg_latency = np.mean(self.results["latencies"]) if self.results["latencies"] else 0
        
        logging.info(f"\n===== Experiment complete: {experiment_name} =====")
        logging.info(f"Successful requests: {len(self.results['latencies'])}/{request_count}")
        logging.info(f"Overall throughput: {overall_throughput:.2f} requests/second")
        logging.info(f"Average latency: {avg_latency:.2f} seconds")
        
        save_results(self.results, f"{results_dir}/{experiment_name}_results.json")
        
        summary = {
            "experiment_name": experiment_name,
            "num_requests": request_count,
            "successful_requests": len(self.results["latencies"]),
            "overall_throughput": overall_throughput,
            "average_latency": avg_latency
        }
        return summary

def setup_logging_and_results(experiment_name: str):
    """Setup logging and results directory for an experiment"""
    # Set up logging
    log_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"task1_{experiment_name}_{log_timestamp}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True  # Force reconfiguration of logging
    )
    logger = logging.getLogger(__name__)
    
    # Get the current cache size from configuration.yaml
    try:
        import yaml
        with open("../../configuration.yaml", 'r') as f:
            config = yaml.safe_load(f)
            current_cache_size = config.get('max_local_cache_size', 'unknown')
    except Exception as e:
        logging.error(f"Error reading configuration.yaml: {e}")
        current_cache_size = "current"
    
    logging.info(f"\n===== Running Task 1 {experiment_name} with cache size: {current_cache_size} =====")
    
    # Create results directory with cache size subfolder
    os.makedirs("task1_results", exist_ok=True)
    cache_results_dir = f"task1_results/{current_cache_size}"
    os.makedirs(cache_results_dir, exist_ok=True)
    logging.info(f"Results will be saved to: {cache_results_dir}")
    
    return cache_results_dir, log_filename, current_cache_size
