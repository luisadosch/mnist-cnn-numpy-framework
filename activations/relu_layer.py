from layer.layer_interface import LayerInterface
from tensor import Tensor
import numpy as np
from typing import List


class ActivationsReLULayer(LayerInterface):

    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):

        for in_tensor, out_tensor in zip(in_tensors, out_tensors):
            # ReLU anwenden: ReLu(x) = max(0, x) -> neg. Werte werden zu 0
            out_tensor.elements = np.maximum(0, in_tensor.elements)

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):

        for in_tensor, out_tensor in zip(in_tensors, out_tensors):
            # Fehler vom nächsten Layer
            delta_y = out_tensor.deltas
            # Ableitung der ReLU-Funktion -> Numpy führt Elementweise Ableitung aus
            relu_grad = np.where(in_tensor.elements > 0, 1.0, 0.0)
            # Elementweise Multiplikation
            delta_x = delta_y * relu_grad
            # ins in_tensors schreiben
            in_tensor.deltas = delta_x
