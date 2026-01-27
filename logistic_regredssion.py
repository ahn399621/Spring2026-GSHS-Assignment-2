import numpy as np
from sklearn.linear_model import LogisticRegression

def logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray
) -> np.ndarray:
    """
    Implements Logistic Regression for binary classification.

    Parameters:
        x_train: numpy.ndarray of shape (num_train, 2)
        y_train: numpy.ndarray of shape (num_train,)
        x_test:  numpy.ndarray of shape (num_test, 2)

    Returns:
        y_pred: numpy.ndarray of shape (num_test,)
    """

    # Logistic Regression 모델 생성
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000
    )

    # 모델 학습
    model.fit(x_train, y_train)

    # 테스트 데이터 예측
    y_pred = model.predict(x_test)

    return y_pred
