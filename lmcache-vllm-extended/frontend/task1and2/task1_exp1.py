
"""
Task 1 - Experiment 1: Latency vs Sequence Length
"""

import datetime
from task1_base import Task1RequestGenerator, setup_logging_and_results, IP, PORT
from tools import generate_requests, plot_sequence_length_vs_latency, analyze_results_by_context, plot_context_comparison

def main():
    cache_results_dir, log_filename, current_cache_size = setup_logging_and_results("exp1_sequence_length")
    generator = Task1RequestGenerator(ip=IP, port=PORT)
    generator.load_contexts("../data")

    questions = [
        "What is the main topic of this document?"
    ]
    
    print(f"\n===== Experiment 1: Latency vs Sequence Length =====")
    print(f"Cache size: {current_cache_size}")
    
    # Run experiment
    exp1_generator = generate_requests(generator.contexts, questions)
    exp1_summary = generator.run_experiment_with_generator(
        exp1_generator, 
        "exp1_sequence_length", 
        max_requests=10, 
        results_dir=cache_results_dir
    )
    
    # Generate plots
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot sequence length vs latency
    plot_sequence_length_vs_latency(
        generator.results, 
        f"{cache_results_dir}/Exp1_sequence_length_vs_latency_{timestamp}.png"
    )
    
    # Analyze and plot context comparison
    context_analysis = analyze_results_by_context(generator.results)
    plot_context_comparison(
        context_analysis, 
        "latency", 
        f"{cache_results_dir}/Exp1_context_comparison_{timestamp}.png"
    )
    
    # Print summary
    print(f"\n===== Experiment 1 Complete =====")
    print(f"Experiment: {exp1_summary['experiment_name']}")
    print(f"Total requests: {exp1_summary['num_requests']}")
    print(f"Successful requests: {exp1_summary['successful_requests']}")
    print(f"Overall throughput: {exp1_summary['overall_throughput']:.2f} requests/second")
    print(f"Average latency: {exp1_summary['average_latency']:.2f} seconds")

if __name__ == "__main__":
    main()
