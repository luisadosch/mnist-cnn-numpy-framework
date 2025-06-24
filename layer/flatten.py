from layer.layer_interface import LayerInterface
from tensor import Tensor, Shape
import json


class Flatten(LayerInterface):
    def __init__(self, in_shape: Shape, out_shape: Shape):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape

    def forward(self, in_tensors: list[Tensor], out_tensors: list[Tensor]):
        for in_tensor, out_tensor in zip(in_tensors, out_tensors):
            out_tensor.elements = in_tensor.elements.reshape(self.out_shape.axis[0], )

    def backward(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        for out_tensor, in_tensor in zip(out_tensors, in_tensors):
            # in_tensor = Tensor(in_tensor.elements.reshape(self.in_shape.axis[0], self.in_shape.axis[1], 1))
            in_tensor.elements = out_tensor.elements.reshape(self.in_shape.axis[0], self.in_shape.axis[1],
                                                             self.in_shape.axis[2])
            in_tensor.shape = self.in_shape
            in_tensor.deltas = out_tensor.deltas.reshape(
                (self.in_shape.axis[0], self.in_shape.axis[1], self.in_shape.axis[2]))

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