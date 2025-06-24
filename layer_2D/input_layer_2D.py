from layer.layer_interface import LayerInterface
from tensor import Tensor
import numpy.typing as npt


class InputLayer2D[T](LayerInterface):
    def __init__(self):
        super().__init__()

    def forward[T](self, data: list[npt.NDArray]) -> list[Tensor]:
        highest_value = 255
        data_tensor_list = [Tensor(img.reshape(28, 28, 1) / highest_value) for img in data]

        return data_tensor_list
