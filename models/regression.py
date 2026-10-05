import time
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def run_gd(X_train, X_test, y_train, y_test, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    start_time = time.time()
    
    X_train_gd = np.column_stack([np.ones(len(X_train)), X_train])
    X_test_gd = np.column_stack([np.ones(len(X_test)), X_test])
    
    n_features = X_train_gd.shape[1]
    theta = np.zeros(n_features)
    
    prev_loss = float('inf')
    loss_history = []
    
    for iteration in range(max_iterations):
        y_pred_train = X_train_gd @ theta
        
        if np.any(np.isnan(y_pred_train)) or np.any(np.isinf(y_pred_train)):
            print(f"Numerical instability detected at iteration {iteration}")
            return {"error": "Numerical instability in gradient descent"}
        
        error = y_pred_train - y_train
        
        gradients = (1 / len(X_train_gd)) * (X_train_gd.T @ error)
        
        gradients = np.clip(gradients, -1e6, 1e6)
        
        theta = theta - learning_rate * gradients
        
        loss = np.mean(error ** 2)
        loss_history.append(loss)
        
        if abs(prev_loss - loss) < tolerance:
            break
            
        prev_loss = loss
    
    preds = X_test_gd @ theta
    
    if np.any(np.isnan(preds)):
        print("Warning: NaN values in predictions")
        return {"error": "NaN values in predictions"}
    
    end_time = time.time()
    
    return {
        "r2": r2_score(y_test, preds),
        "mse": mean_squared_error(y_test, preds),
        "iterations": iteration + 1,
        "preds": preds,
        "y_test": y_test,
        "time": end_time - start_time,
        "loss": loss_history
    }

if __name__ == "__main__":
    from data_utils import load_data
    X_train, X_test, y_train, y_test = load_data()
    result = run_gd(X_train, X_test, y_train, y_test, 0.01)
    print("GD Result (LR=0.01):", result)

