
"""
Task 1 - Experiment 3b: Impact of Context Diversity - High Diversity
"""

import datetime
from task1_base import Task1RequestGenerator, setup_logging_and_results, IP, PORT
from tools import generate_requests_shuffle, plot_request_number_vs_latency, plot_throughput_over_time

def main():
    cache_results_dir,current_cache_size = setup_logging_and_results("exp3b_high_diversity")
    generator = Task1RequestGenerator(ip=IP, port=PORT)
    generator.load_contexts("../data")

    questions = [
        "What is the main topic of this document?",
        "Summarise the main ideas of the method in the document."
    ]
    
    print(f"\n===== Experiment 3b: High Diversity Context =====")
    print(f"Cache size: {current_cache_size}")
    
    # Use all available contexts with shuffling for minimum cache reuse
    if len(generator.contexts) >= 3:
        high_diversity_contexts = generator.contexts  # use all contexts for high diversity
        
        print(f"Using {len(high_diversity_contexts)} contexts for high diversity:")
        for context_id in list(high_diversity_contexts.keys())[:15]: # load all contexts but only print first 15
            print(f"  - {context_id}")
        if len(high_diversity_contexts) > 5:
            print(f"  ... and {len(high_diversity_contexts) - 5} more contexts")
        
        print("Testing: Minimum cache reuse with maximum context variety and shuffling")
        
        # generate_requests_shuffle 生成请求并打乱顺序
        high_diversity_generator = generate_requests_shuffle(high_diversity_contexts, questions)
        
        # Run the experiment
        exp3b_summary = generator.run_experiment_with_generator(
            high_diversity_generator, 
            "exp3b_high_diversity", 
            max_requests=10, 
            results_dir=cache_results_dir
        )
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_request_number_vs_latency(
            generator.results, 
            f"{cache_results_dir}/Exp3b_high_diversity_latency_{timestamp}.png"
        )
        plot_throughput_over_time(
            generator.results, 
            f"{cache_results_dir}/Exp3b_high_diversity_throughput_{timestamp}.png"
        )
        
        # Print summary
        print(f"\n===== Experiment 3b Complete =====")
        print(f"Experiment: {exp3b_summary['experiment_name']}")
        print(f"Total contexts available: {len(high_diversity_contexts)}")
        print(f"Total requests: {exp3b_summary['num_requests']}")
        print(f"Successful requests: {exp3b_summary['successful_requests']}")
        print(f"Overall throughput: {exp3b_summary['overall_throughput']:.2f} requests/second")
        print(f"Average latency: {exp3b_summary['average_latency']:.2f} seconds")
        
    else:
        print("Error: Need at least 3 contexts for high diversity test!")

if __name__ == "__main__":
    main()
