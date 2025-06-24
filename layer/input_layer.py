from layer.layer_interface import LayerInterface
from tensor import Tensor
import numpy.typing as npt

class InputLayer[T](LayerInterface):
    def __init__(self):
        super().__init__()

    def forward[T](self, data: list[npt.NDArray]) -> list[Tensor]:
        highest_value = 255
        data_tensor_list = [Tensor(elements/highest_value) for elements in data]
        return data_tensor_list

