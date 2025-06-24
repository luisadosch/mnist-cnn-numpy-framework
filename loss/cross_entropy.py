import numpy as np
from loss.loss_interface import LossInterface
from tensor import Tensor
from typing import List

class CrossEntropy(LossInterface):

    def __init__(self):
        super().__init__()

    def forward(self, predictions: List[Tensor], labels: List[Tensor]) -> float:
        """
        Compute the total cross-entropy loss:
            L = - Σ_i Σ_j (t_{i,j} * log(x_{i,j}))
        where:
          - i indexes over samples / tensors
          - j indexes over the elements within each tensor
        """
        total_loss = 0.0
        for pred_tensor, label_tensor in zip(predictions, labels):
            # For numerical stability, clip predictions away from 0 (log(0) is undefined)
            x = np.clip(pred_tensor.elements, 1e-12, 1.0)
            t = label_tensor.elements
            total_loss -= np.sum(t * np.log(x))
        return total_loss

    def backward(self, predictions: List[Tensor], labels: List[Tensor]):
        """
        Compute the gradient of the loss w.r.t. the predictions:
            ∂L/∂x_{i,j} = - t_{i,j} / x_{i,j}
        Stores the result in predictions[*].deltas and returns the list of prediction tensors.
        """
        for pred_tensor, label_tensor in zip(predictions, labels):
            # Clip again to match forward numerical stability
            x = np.clip(pred_tensor.elements, 1e-12, 1.0)
            t = label_tensor.elements
            grad = - t / x
            pred_tensor.deltas = grad