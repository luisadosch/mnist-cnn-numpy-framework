from layer.layer_interface import LayerInterface
from tensor import Tensor, Shape
from typing import List
import numpy as np
import json


class FullyConnectedLayer(LayerInterface):

    def __init__(self, in_shape: Shape, out_shape: Shape):
        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

        self.weights = Tensor(elements=np.random.randn(in_shape.axis[0], out_shape.axis[0]) * np.sqrt(2. / in_shape.axis[0]))
        self.bias = Tensor(elements=np.zeros(out_shape.axis[0]))
        """        
        self.weights = Tensor(elements=np.random.uniform(low=-0.5, high=0.5, size=(in_shape.axis[0], out_shape.axis[0])))
        self.bias = Tensor(elements=np.random.uniform(low=-0.5, high=0.5, size=(out_shape.axis[0],)))
        """

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        for input_tensor, output_tensor in zip(in_tensors, out_tensors):
            # @ macht Matrixmultiplikation
            output_tensor.elements[:] = input_tensor.elements @ self.weights.elements + self.bias.elements

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        for output_tensor, input_tensor in zip(out_tensors, in_tensors):
            input_tensor.deltas[:] = self.weights.elements @ output_tensor.deltas

    def calculate_delta_weights(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        # Join input and output tensor into 2D arrays eine Achse Batch eine elements
        inputs = np.stack([t.elements for t in in_tensors])
        deltas = np.stack([t.deltas for t in out_tensors])

        # weight and delta per single data instead of whole batch
        batch_size = len(in_tensors)
        weight_deltas = inputs.T @ deltas / batch_size
        bias_deltas = deltas.mean(axis=0)

        # [:] ändert Inhalt den Inhalt des Arrays deltas
        # (ohne [:] würde nur die Referenz geändert werden,
        # wenn an anderen Stellen noch eine Referenz auf das alte besteht ist das problematisch)
        self.weights.deltas[:] = weight_deltas
        self.bias.deltas[:] = bias_deltas

    def initialize_weights_bias(self, weights: Tensor, bias: Tensor):
        self.weights = weights
        self.bias = bias

    def save_params(self, path):
        data = {
            "weights": self.weights.elements.tolist(),
            "bias": self.bias.elements.tolist()
        }
        with open(path+".json", "w") as file:
            json.dump(data, file)

    def load_params(self, path):
        with open(path, "r") as openfile:
            json_data = json.load(openfile)
            self.weights = Tensor(np.array(json_data["weights"]))
            self.bias = Tensor(np.array(json_data["bias"]))

    def to_dict(self):
        return {
            "type": str(type(self)),
            "in_shape": self.in_shape.axis,
            "out_shape": self.out_shape.axis
        }

    @classmethod
    def from_dict(cls, data):
        layer = cls(in_shape=Shape(axis=data["in_shape"]), out_shape=Shape(axis=data["out_shape"]))
        return layer

    def forward_old(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        for x in range(len(in_tensors)):
            for i in range(out_tensors[x].shape.axis[0]):
                result = 0
                for j in range(in_tensors[x].shape.axis[0]):
                    result += in_tensors[x].elements[j] * self.weights.elements[j][i]
                result += self.bias.elements[i]
                out_tensors[x].elements[i] = result

    def backward_old(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        for x in range(len(out_tensors)):
            for i in range(in_tensors[x].shape.axis[0]):
                delta_sum = 0
                for j in range(out_tensors[x].shape.axis[0]):
                    delta_sum += out_tensors[x].deltas[j] * self.weights.elements[i][j]

                in_tensors[x].deltas[i] = delta_sum

    def calculate_delta_weights_old(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        for x in range(len(out_tensors)):
            for i in range(out_tensors[x].shape.axis[0]):
                self.bias.deltas[i] += out_tensors[x].deltas[i]
                for j in range(in_tensors[x].shape.axis[0]):
                    self.weights.deltas[j][i] = out_tensors[x].deltas[i] * in_tensors[x].elements[j]
        self.weights.deltas /= len(out_tensors)
        self.bias.deltas /= len(out_tensors)