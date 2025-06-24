from layer.layer_interface import LayerInterface
import numpy as np
from tensor import Tensor
from typing import List


class ActivationsSigmoidLayer(LayerInterface):

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        """
        Forward-Pass:
        y = sigmoid(x) = 1 / (1 + exp(-x))
        """
        for in_tensor, out_tensor in zip(in_tensors, out_tensors):
            # Sigmoid anwenden:out_tensor =  sigmoid(x) = 1 / (1 + exp(-x))
            out_tensor.elements = 1.0 / (1.0 + np.exp(-in_tensor.elements))
            """
            Todo: Prof. wegen RuntimeWarning frage. Ob es Auswirkungen aus unser Netzwerk hat.
            
            -708 > Werte > 709 Daher entweder 0 oder Inf
            """
            #print(np.sort(in_tensor.elements))
            #sys.exit()


    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        """
        Backward-Pass:
        δX = δY ⊙ sigmoid(x) ⊙ (1 - sigmoid(x))
        """

        for in_tensor, out_tensor in zip(in_tensors, out_tensors):
            delta_y = out_tensor.deltas

            # Ableitung der Sigmoid-Funktion
            sig_grad = out_tensor.elements * (1.0 - out_tensor.elements)

            # Elementweise Multiplikation
            delta_x = delta_y * sig_grad

            # ins in_tensors schreiben
            in_tensor.deltas = delta_x
