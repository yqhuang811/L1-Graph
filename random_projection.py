import numpy as np
from scipy.linalg import qr

def random_projection(X, k):
    n, d = X.shape
    Z = np.zeros((n, n))
    
    # Create random projection matrix Ω
    Omega = np.random.randn(d, k)

    # QR decomposition of XΩ
    X_Omega = X @ Omega
    Q, R = qr(X_Omega, mode='economic')
    
    # Project X onto lower-dimensional space
    X_tilde = Q @ (Q.T @ X)

    return X_tilde
