import vllm.entrypoints.openai.api_server as base_api
from vllm.entrypoints.openai.protocol import *
from fastapi import APIRouter, Request, Body
from typing import List, Dict, Any, Optional
from collections import defaultdict
import time
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("batch_scheduler")

# Router for the extended API
extended_router = APIRouter()

# Global variables to track request batches and contexts
request_batches = []
context_tracking = {}  # Maps request_id to context_id
performance_metrics = {
    "batch_sizes": [],
    "processing_times": [],
    "throughputs": [],
    "context_switches": []
}

@extended_router.get("/models")
async def show_available_models(request: Request):
    logger.info("v2 models is called!")
    return await base_api.show_available_models(request)

@extended_router.post("/batch/chat/completions")
async def create_batch_chat_completion(
    requests: List[Dict[str, Any]] = Body(...),
    raw_request: Request = None
):
    """
    Process a batch of chat completion requests.
    This endpoint accepts a list of chat completion requests and processes them as a batch.
    
    Args:
        requests: List of chat completion request objects
        raw_request: The raw HTTP request
        
    Returns:
        List of chat completion responses
    """
    logger.info("\t\t-- CREATE_BATCH_CHAT_COMPLETION CALLED --")
    batch_start_time = time.time()
    batch_size = len(requests)
    logger.info(f"Received batch of {batch_size} requests")
    
    # Extract context information from each request
    request_contexts = []
    for i, req in enumerate(requests):
        # Extract context from the request
        # Assuming the context is in the first user message
        if "messages" in req and len(req["messages"]) > 0:
            for msg in req["messages"]:
                if msg["role"] == "user" and "content" in msg:
                    content = msg["content"]
                    if content.startswith("Context:"):
                        # Extract context ID if provided in the request
                        context_id = req.get("context_id", f"context_{i}")
                        request_contexts.append({
                            "request_index": i,
                            "context_id": context_id
                        })
                        break
    
    # Group requests by context
    context_groups = defaultdict(list)
    for item in request_contexts:
        context_groups[item["context_id"]].append(item["request_index"])
    
    # Create a reordered list of request indices
    scheduled_indices = []
    for context_id, indices in context_groups.items():
        scheduled_indices.extend(indices)
    
    # Count context switches
    context_switches = 0
    prev_context = None
    for i, idx in enumerate(scheduled_indices):
        current_context = request_contexts[idx]["context_id"]
        if i > 0 and current_context != prev_context:
            context_switches += 1
        prev_context = current_context
    
    # Reorder the requests based on context grouping
    scheduled_requests = [requests[idx] for idx in scheduled_indices]
    
    # Process each request sequentially
    responses = []
    for req in scheduled_requests:
        # Convert dict to ChatCompletionRequest
        chat_req = ChatCompletionRequest(**req)
        # Process the request
        response = await base_api.create_chat_completion(chat_req, raw_request)
        responses.append(response)
    
    # Calculate metrics
    batch_end_time = time.time()
    processing_time = batch_end_time - batch_start_time
    throughput = batch_size / processing_time if processing_time > 0 else 0
    
    # Store metrics
    performance_metrics["batch_sizes"].append(batch_size)
    performance_metrics["processing_times"].append(processing_time)
    performance_metrics["throughputs"].append(throughput)
    performance_metrics["context_switches"].append(context_switches)
    
    logger.info(f"Batch processed in {processing_time:.2f}s, throughput: {throughput:.2f} req/s, context switches: {context_switches}")
    
    return responses

@extended_router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    """
    Process a single chat completion request.
    This is the standard endpoint that processes one request at a time.
    """
    logger.info("-------------- V2 COMPLETION IS CALLED FOR A SINGLE REQUEST")
    return await base_api.create_chat_completion(request, raw_request)

@extended_router.get("/metrics")
async def get_performance_metrics():
    """
    Get performance metrics for the batch scheduler.
    """
    avg_throughput = sum(performance_metrics["throughputs"]) / len(performance_metrics["throughputs"]) if performance_metrics["throughputs"] else 0
    avg_processing_time = sum(performance_metrics["processing_times"]) / len(performance_metrics["processing_times"]) if performance_metrics["processing_times"] else 0
    avg_context_switches = sum(performance_metrics["context_switches"]) / len(performance_metrics["context_switches"]) if performance_metrics["context_switches"] else 0
    
    return {
        "total_batches_processed": len(performance_metrics["batch_sizes"]),
        "total_requests_processed": sum(performance_metrics["batch_sizes"]),
        "average_batch_size": sum(performance_metrics["batch_sizes"]) / len(performance_metrics["batch_sizes"]) if performance_metrics["batch_sizes"] else 0,
        "average_throughput": avg_throughput,
        "average_processing_time": avg_processing_time,
        "average_context_switches": avg_context_switches,
        "raw_metrics": performance_metrics
    }

@extended_router.post("/reset_metrics")
async def reset_performance_metrics():
    """
    Reset performance metrics
    """
    global performance_metrics
    performance_metrics = {
        "batch_sizes": [],
        "processing_times": [],
        "throughputs": [],
        "context_switches": []
    }
    return {"status": "success", "message": "Performance metrics have been reset"}
