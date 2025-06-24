from layer.layer_interface import LayerInterface
import numpy as np
from tensor import Tensor
from typing import List


class ActivationsTanhLayer(LayerInterface):

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        """
        Forward-Pass:
        y = tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        """

        for in_tensor, out_tensor in zip(in_tensors, out_tensors):
            # Tanh anwenden: tanh(x) = (e^x - e^-x) / (e^x + e^-x)

            out_tensor.elements = np.tanh(in_tensor.elements)

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        """
        Backward-Pass:
        δX = δY ⊙ (1 - tanh(x)^2)
        """

        for in_tensor, out_tensor in zip(in_tensors, out_tensors):
            delta_y = out_tensor.deltas

            tanh_grad = 1.0 - out_tensor.elements ** 2
            delta_x = delta_y * tanh_grad
            in_tensor.deltas = delta_x
