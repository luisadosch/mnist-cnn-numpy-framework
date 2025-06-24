import numpy as np
from loss.loss_interface import LossInterface
from tensor import Tensor
from typing import List

class MeanSquaredError(LossInterface):

    def __init__(self):
        super().__init__()

    def forward(self, predictions: List[Tensor], labels: List[Tensor]) -> float:
        """
        Compute the mean squared error over all tensors and their elements:
            L = (1 / N) * Σ_i Σ_j (x_{i,j} - t_{i,j})^2
        where N = total number of elements across all tensors.
        """
        total_loss = 0.0
        total_elements = 0

        for pred_tensor, label_tensor in zip(predictions, labels):
            x = pred_tensor.elements
            t = label_tensor.elements
            # accumulate squared error
            total_loss += np.sum((x - t) ** 2)
            total_elements += x.size

        # return the mean over all elements
        return total_loss / total_elements

    def backward(self, predictions: List[Tensor], labels: List[Tensor]):
        """
        Compute the gradient of MSE w.r.t. each prediction:
            ∂L/∂x_{i,j} = (2 / N) * (x_{i,j} - t_{i,j})
        Stores each gradient in predictions[*].deltas and returns predictions.
        """
        # first compute total number of elements to keep N consistent
        total_elements = sum(pred.elements.size for pred in predictions)

        for pred_tensor, label_tensor in zip(predictions, labels):
            x = pred_tensor.elements
            t = label_tensor.elements
            grad = (2.0 / total_elements) * (x - t)
            pred_tensor.deltas = grad
