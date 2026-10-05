import pandas as pd

def combine_results():
    # Read the CSV files
    linear_results = pd.read_csv('../Linear-Regression/results.csv')
    neural_results = pd.read_csv('../Neural-network/results_nn.csv')
    
    # Combine the dataframes
    combined_results = pd.concat([linear_results, neural_results])
    
    # Remove duplicates based on all columns
    combined_results = combined_results.drop_duplicates()
    
    # Sort by Model and Learning Rate
    combined_results = combined_results.sort_values(['Model', 'Learning Rate'])
    
    # Reset the index
    combined_results = combined_results.reset_index(drop=True)
    
    # Save the combined results
    combined_results.to_csv('combined_results.csv', index=False)
    
    return combined_results

if __name__ == "__main__":
    results = combine_results()
    print("Combined results saved to 'combined_results.csv'")
    print(results)