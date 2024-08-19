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
