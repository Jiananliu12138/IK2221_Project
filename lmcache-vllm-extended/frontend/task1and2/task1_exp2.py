
"""
Task 1 - Experiment 2: Cache Reuse with Repeated Context
"""

import datetime
from task1_base import Task1RequestGenerator, setup_logging_and_results, IP, PORT
from tools import generate_requests, plot_request_number_vs_latency

def main():
    cache_results_dir, current_cache_size = setup_logging_and_results("exp2_cache_reuse")
    generator = Task1RequestGenerator(ip=IP, port=PORT)
    generator.load_contexts("../data")

    questions = [
        "What is the main topic of this document?",
        "Summarise the main ideas of the method in the document.",
        "What is the research context of this paper?",
        "How the main experiments in this paper were conducted?",
        "What exactly are the experimental results obtained from this question?"
        "What are the main conclusions in this document?"
    ]
    
    print(f"\n===== Experiment 2: Cache Reuse with Repeated Context =====")
    print(f"Cache size: {current_cache_size}")
    
    # Select a single context for repeated use
    if len(generator.contexts) > 0:
        context_id = list(generator.contexts.keys())[1]  # 报告里跑了两个不同的context
        single_context = {context_id: generator.contexts[context_id]}
        
        print(f"Using context: {context_id}")
        print("Testing: Would the response be faster if I asked the same questions consecutively for the same document?")
        
        repeated_requests = []
        for round_num in range(2):  # Run the same questions twice
            print(f"Generating requests for round {round_num + 1}")
            repeated_requests.extend(generate_requests(single_context, questions))
        
        print(f"Total requests to be sent: {len(repeated_requests)}")
        
        # Run the experiment
        exp2_summary = generator.run_experiment_with_generator(
            iter(repeated_requests),  # iter 逐条拉取请求并发送(per question)
            "exp2_cache_reuse", 
            results_dir=cache_results_dir
        )
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_request_number_vs_latency(
            generator.results, 
            f"{cache_results_dir}/Exp2_request_number_vs_latency_{timestamp}.png"
        )
        
        print(f"\n===== Experiment 2 Complete =====")
        print(f"Experiment: {exp2_summary['experiment_name']}")
        print(f"Context used: {context_id}")
        print(f"Total requests: {exp2_summary['num_requests']}")
        print(f"Successful requests: {exp2_summary['successful_requests']}")
        print(f"Overall throughput: {exp2_summary['overall_throughput']:.2f} requests/second")
        print(f"Average latency: {exp2_summary['average_latency']:.2f} seconds")

        # Analyze cache reuse effect
        if len(generator.results["latencies"]) >= 12:  # 6 questions * 2 rounds
            first_round_latencies = generator.results["latencies"][:6]
            second_round_latencies = generator.results["latencies"][6:12]
            
            avg_first_round = sum(first_round_latencies) / len(first_round_latencies)
            avg_second_round = sum(second_round_latencies) / len(second_round_latencies)
            
            print(f"\n===== Cache Reuse Analysis =====")
            print(f"First round average latency: {avg_first_round:.2f}s")
            print(f"Second round average latency: {avg_second_round:.2f}s")
    else:
        print("Error: No contexts available for testing!")

if __name__ == "__main__":
    main()
