from layer.layer_interface import LayerInterface
import numpy as np
from tensor import Tensor
from typing import List


class SoftmaxOutputLayer(LayerInterface):

    def __init__(self):
        super().__init__()

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        """
        Forward-Pass:
          y_i = exp(x_i) / sum_j exp(x_j)
        numerisch stabilisiert durch Subtraktion des Max-Werts pro Zeile.
        """

        for in_tensor, out_tensor in zip(in_tensors, out_tensors):
            shifted_elements = in_tensor.elements - np.max(in_tensor.elements)
            exp_elements = np.exp(shifted_elements)
            out_tensor.elements = exp_elements / np.sum(exp_elements)

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        """
        Backward-Pass über die Softmax-Jacobian:
          δX[k] = J[k] @ δY[k]  für jede Sample-Zeile k
          J = diag(y) - y y^T
        """

        for in_tensor, out_tensor in zip(in_tensors, out_tensors):
            y = out_tensor.elements.reshape(-1, 1)  # Softmax output
            dy = out_tensor.deltas.reshape(-1, 1)  # Gradient from next layer

            jacobian = np.diagflat(y) - np.dot(y, y.T)
            dx = (jacobian @ dy).flatten()

            in_tensor.deltas = dx
