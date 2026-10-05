from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def combine_results():
    # Read the CSV files
    linear_results = pd.read_csv(PROJECT_ROOT / 'Linear-Regression' / 'results.csv')
    neural_results = pd.read_csv(PROJECT_ROOT / 'Neural-network' / 'results_nn.csv')
    
    # Combine the DataFrames
    combined_results = pd.concat([linear_results, neural_results])
    
    # Remove duplicates based on all columns
    combined_results = combined_results.drop_duplicates()
    
    # Sort by Model and Learning Rate
    combined_results = combined_results.sort_values(['Model', 'Learning Rate'])
    
    # Reset the index
    combined_results = combined_results.reset_index(drop=True)
    
    # Save the combined results
    combined_results.to_csv(Path(__file__).resolve().parent / 'combined_results.csv', index=False)
    
    return combined_results

if __name__ == "__main__":
    results = combine_results()
    print("Combined results saved to 'combined_results.csv'")
    print(results)
