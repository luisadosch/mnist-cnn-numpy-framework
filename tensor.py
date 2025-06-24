from typing import List
import numpy as np
import numpy.typing as npt

global float_class


class Shape:  # Shape with axis: [3,3] refers to a matrix with 3 rows and 3 columns
    def __init__(self, axis: List[int]):
        self.axis = axis
        # /volume # Int


class Tensor:
    float_class = np.float64

    def __init__(self, elements: npt.NDArray[float_class]):
        self.elements = elements  # data of tensor
        self.shape = Shape(axis=elements.shape)
        self.deltas = np.zeros_like(elements)  # deltas for backpropagation

    @classmethod
    def from_shape(cls, shape: Shape):
        return Tensor(elements=np.zeros(shape.axis, dtype=Tensor.float_class))

    @classmethod
    def empty_like(cls, tensor: 'Tensor'):
        return cls(np.empty_like(tensor.elements))
