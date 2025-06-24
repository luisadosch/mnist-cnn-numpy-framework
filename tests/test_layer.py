from layer.fully_connected_layer import FullyConnectedLayer
import unittest
import numpy as np
from tensor import Tensor, Shape

class TestFullyConnectedLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.weight_matrix = Tensor(
            elements=np.array([[3, 5], [4, 6]], dtype=np.float64)
        )
        self.bias = Tensor(elements=np.array([0.5, 0.6], dtype=np.float64))
        self.fc_layer = FullyConnectedLayer(in_shape=Shape((2,1)), out_shape=Shape((2, 1)))
        self.fc_layer.weights = self.weight_matrix
        self.fc_layer.bias = self.bias

    def test_forward(self) -> None:
        in_tensors = [Tensor(elements=np.array([1, 2], dtype=np.float64))]
        out_tensors = [Tensor(elements=np.array([0, 0], dtype=np.float64))]
        self.fc_layer.forward(in_tensors, out_tensors)
        self.assertTrue(
            np.array_equal(out_tensors[0].elements, np.array([11.5, 17.6])),
            "FC Layer forward function does not calculate the correct outputs",
        )

    def test_backward(self) -> None:
        in_tensors = [Tensor(elements=np.array([1, 2], dtype=np.float64))]
        out_tensors = [Tensor(elements=np.array([0, 0], dtype=np.float64))]
        out_tensors[0].deltas = np.array([8, 9])
        self.fc_layer.backward(out_tensors, in_tensors)
        self.assertTrue(
            np.array_equal(in_tensors[0].deltas, np.array([69, 86])),
            "FC Layer backward function does not calculate the correct outputs",
        )

    def test_calculate_deltas(self) -> None:
        in_tensors = [Tensor(elements=np.array([1, 2], dtype=np.float64))]
        out_tensors = [Tensor(elements=np.array([0, 0], dtype=np.float64))]
        out_tensors[0].deltas = np.array([8, 9])
        self.fc_layer.calculate_delta_weights(out_tensors, in_tensors)
        self.assertTrue(
            np.array_equal(self.weight_matrix.deltas, np.array([[8, 9], [16, 18]])),
            "FCLayer calculate delta weights function does not calculate the correct deltas for the weight matrix",
        )
        self.assertTrue(
            np.array_equal(self.bias.deltas, out_tensors[0].deltas),
            "calculate delta weights function does not calculate the correct deltas for the bias",
        )



if __name__ == "__main__":
    unittest.main()
