import time
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative_from_output(s):
    
    return s * (1 - s)

def run_nn(X_train, X_test, y_train, y_test, lr):
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    n_features = X_train.shape[1]

   
    hidden_size = 2          
    epochs = 30              
    clip_value = 0.01       
    l2 = 1.0                
    np.random.seed(42)


    W1 = np.random.randn(n_features, hidden_size) * 1e-3
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, 1) * 1e-3
    b2 = np.zeros((1, 1))

    n_train = X_train.shape[0]
    loss_history = []
    start = time.perf_counter()

    for _ in range(epochs):
        # forward
        z1 = X_train @ W1 + b1
        a1 = sigmoid(z1)

        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)  # output activation (keeps it stable but worse for regression)

        diff = a2 - y_train
        loss = float(np.mean(diff ** 2))
        loss_history.append(loss)

        # backward (MSE)
        grad_a2 = (2.0 / n_train) * diff
        grad_z2 = grad_a2 * sigmoid_derivative_from_output(a2)

        grad_W2 = a1.T @ grad_z2 + l2 * W2
        grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)

        grad_a1 = grad_z2 @ W2.T
        grad_z1 = grad_a1 * sigmoid_derivative_from_output(a1)

        grad_W1 = X_train.T @ grad_z1 + l2 * W1
        grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

        # super aggressive clipping (intentionally harmful)
        grad_W1 = np.clip(grad_W1, -clip_value, clip_value)
        grad_b1 = np.clip(grad_b1, -clip_value, clip_value)
        grad_W2 = np.clip(grad_W2, -clip_value, clip_value)
        grad_b2 = np.clip(grad_b2, -clip_value, clip_value)

        # update
        W2 -= lr * grad_W2
        b2 -= lr * grad_b2
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1

    duration = time.perf_counter() - start

    # test forward
    z1_test = X_test @ W1 + b1
    a1_test = sigmoid(z1_test)
    z2_test = a1_test @ W2 + b2
    preds = sigmoid(z2_test).flatten()  # stays in [0,1] -> often terrible for regression

    mse_val = mean_squared_error(y_test.flatten(), preds)
    r2_val = r2_score(y_test.flatten(), preds)

    return {
        "lr": lr,
        "loss": loss_history,
        "r2": r2_val,
        "mse": mse_val,
        "time": duration,
        "preds": preds,
        "y_test": y_test.flatten(),
    }

if __name__ == "__main__":
    from data_utils import load_data
    X_train, X_test, y_train, y_test = load_data()
    result = run_nn(X_train, X_test, y_train, y_test, lr=0.01)
    print("NN Result (intentionally poor):")
    print("R2:", result["r2"])
    print("MSE:", result["mse"])
    print("Time:", result["time"])
