import numpy as np

def lasso_gradient_descent(X, y, alpha, learning_rate=0.01, num_iterations=10000, tol=1e-4):
    m, n = X.shape
    coefficients = np.zeros(n)
    
    for iteration in range(num_iterations):
        # Calculate the predictions
        predictions = X @ coefficients
        
        # Calculate the gradient of the squared error part
        gradient = - (1 / m) * (X.T @ (y - predictions))
        
        # Update coefficients with gradient descent step
        coefficients = coefficients - learning_rate * gradient
        
        # Apply the L1 penalty (soft thresholding)
        coefficients = np.sign(coefficients) * np.maximum(np.abs(coefficients) - learning_rate * alpha, 0)
        
        # Check for convergence
        if np.max(np.abs(learning_rate * gradient)) < tol:
            break
    
    return coefficients

def main():
    n_samples = 100
    n_features = 100
    
    # Generate a random X matrix
    X = np.random.randn(n_samples, n_features)
    
    # Initialize Z matrix
    n = X.shape[0]
    Z = np.zeros((n, n))

    # Regularization parameter lambda
    lambda_val = 0.1

    # Solve for each Z^i using Lasso
    for i in range(n):
        target = X[:, i]
        X_mod = np.delete(X, i, axis=1)  # Remove the i-th column to avoid the trivial solution
        Z[i, np.arange(n) != i] = lasso_gradient_descent(X_mod, target, alpha=lambda_val)

    # Print the resulting Z matrix
    print("Resulting Z matrix:")
    print(Z)

if __name__ == "__main__":
    main()
