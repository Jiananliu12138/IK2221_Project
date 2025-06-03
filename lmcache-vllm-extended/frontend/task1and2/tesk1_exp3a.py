
"""
Task 1 - Experiment 3a: Impact of Context Diversity - Low Diversity
"""

import datetime
from task1_base import Task1RequestGenerator, setup_logging_and_results, IP, PORT
from tools import generate_requests, plot_request_number_vs_latency, plot_throughput_over_time

def main():
    cache_results_dir, log_filename, current_cache_size = setup_logging_and_results("exp3a_low_diversity")
    generator = Task1RequestGenerator(ip=IP, port=PORT)
    generator.load_contexts("../data")

    questions = [
        "What is the main topic of this document?",
        "Summarise the main ideas of the method in the document.",
        "What is the research context of this paper?",
        "How the main experiments in this paper were conducted?",
        "What are the main findings of this paper?"
    ]
    
    print(f"\n===== Experiment 3a: Low Diversity Context =====")
    print(f"Cache size: {current_cache_size}")
    
    # Use only 2 contexts for better cache reuse
    if len(generator.contexts) >= 2:
        context_keys = list(generator.contexts.keys())
        low_diversity_contexts = {k: generator.contexts[k] for k in context_keys[:2]}  # use only the first 2 contexts
        
        print(f"Using {len(low_diversity_contexts)} contexts for low diversity:")
        for context_id in low_diversity_contexts.keys():
            print(f"  - {context_id}")
        
        print("Testing: Better cache reuse with limited context variety")
        
        low_diversity_generator = generate_requests(low_diversity_contexts, questions)
        
        # Run the experiment
        exp3a_summary = generator.run_experiment_with_generator(
            low_diversity_generator, 
            "exp3a_low_diversity", 
            max_requests=10, 
            results_dir=cache_results_dir
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_request_number_vs_latency(
            generator.results, 
            f"{cache_results_dir}/Exp3a_low_diversity_latency_{timestamp}.png"
        )
        plot_throughput_over_time(
            generator.results, 
            f"{cache_results_dir}/Exp3a_low_diversity_throughput_{timestamp}.png"
        )
        
        # Print summary
        print(f"\n===== Experiment 3a Complete =====")
        print(f"Experiment: {exp3a_summary['experiment_name']}")
        print(f"Contexts used: {list(low_diversity_contexts.keys())}")
        print(f"Total requests: {exp3a_summary['num_requests']}")
        print(f"Successful requests: {exp3a_summary['successful_requests']}")
        print(f"Overall throughput: {exp3a_summary['overall_throughput']:.2f} requests/second")
        print(f"Average latency: {exp3a_summary['average_latency']:.2f} seconds")
        print(f"Results saved to: {cache_results_dir}")
        print(f"Log file: {log_filename}")
        
    else:
        print("Error: Need at least 2 contexts for low diversity test!")

if __name__ == "__main__":
    main()
