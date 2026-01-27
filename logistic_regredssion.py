import numpy as np

def logistic_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """
    Logistic Regression (binary) using gradient descent.

    Parameters:
        x_train: (n_train, 2)
        y_train: (n_train,) or (n_train, 1) with 0/1
        x_test:  (n_test, 2)

    Returns:
        y_pred: (n_test,) predicted labels (0/1)
    """
    # --- helper functions ---
    def sigmoid(z: np.ndarray) -> np.ndarray:
        # overflow 방지
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    # --- input shaping ---
    X_train = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train, dtype=float).reshape(-1)  # (n,)
    X_test = np.asarray(x_test, dtype=float)

    if X_train.ndim != 2 or X_train.shape[1] != 2:
        raise ValueError("x_train must have shape (n_samples, 2)")
    if X_test.ndim != 2 or X_test.shape[1] != 2:
        raise ValueError("x_test must have shape (n_samples, 2)")
    if y.shape[0] != X_train.shape[0]:
        raise ValueError("y_train length must match x_train rows")

    # --- add bias term ---
    Xb_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]  # (n, 3)
    Xb_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]     # (m, 3)

    # --- train with gradient descent ---
    w = np.zeros(Xb_train.shape[1])  # (3,)
    lr = 0.1
    n_iters = 2000
    eps = 1e-12

    for _ in range(n_iters):
        p = sigmoid(Xb_train @ w)  # (n,)
        # gradient of log-loss
        grad = (Xb_train.T @ (p - y)) / Xb_train.shape[0]
        w -= lr * grad


    # --- predict ---
    p_test = sigmoid(Xb_test @ w)
    y_pred = (p_test >= 0.5).astype(int)  # 0/1
    return y_pred
