import json
import os

from activations.relu_layer import ActivationsReLULayer
from activations.sigmoid_layer import ActivationsSigmoidLayer
from activations.softmax_layer import SoftmaxOutputLayer
from activations.tanh_layer import ActivationsTanhLayer
from layer.flatten import Flatten
from layer.input_layer import InputLayer
from layer.layer_interface import LayerInterface
from layer_2D.pooling_2D_layer import Pooling2DLayer, PoolingType
from tensor import Tensor, Shape
from typing import List
import numpy as np
import numpy.typing as npt
from layer.fully_connected_layer import FullyConnectedLayer
from layer_2D.convolution_layer import Conv2DLayer


class Network:
    def __init__(self, caches: List[List[Tensor]], input_layer: LayerInterface, layers: list[LayerInterface],
                 delta_params: List[Tensor] = [],
                 ):
        self.input_layer = input_layer
        self.layers = layers
        self.delta_params = delta_params
        self.caches = caches

    def forward(self, data: List[npt.NDArray]):
        self.caches = []

        # Input Layer
        in_tensors = self.input_layer.forward(data)
        self.caches.append(in_tensors)
        for layer in self.layers:
            # initialize enough out_tensors for layer
            out_tensors = []
            for _ in in_tensors:
                if hasattr(layer, 'out_shape'):
                    out_tensors.append(Tensor(elements=np.zeros(layer.out_shape.axis, dtype=Tensor.float_class)))
                else:
                    out_tensors.append(Tensor(elements=np.zeros(in_tensors[0].shape.axis[0], dtype=Tensor.float_class)))

            layer.forward(in_tensors, out_tensors)
            self.caches.append(out_tensors)
            # in_tensors von nächstem Layer sind out_tensors dieses Layers
            in_tensors = out_tensors

        # gibt die out_tensoren des letzten Layers zurück
        return out_tensors

    def backprop(self):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            out_tensors = self.caches[i + 1]
            in_tensors = self.caches[i]  # use input if it's the first layer

            layer.backward(out_tensors=out_tensors, in_tensors=in_tensors)

            if isinstance(layer, (FullyConnectedLayer, Conv2DLayer)):
                layer.calculate_delta_weights(out_tensors, in_tensors)

    def save_params(self, folder_path, name):
        os.makedirs(folder_path, exist_ok=True)
        path = os.path.join(folder_path, name)
        layer_data = [layer.to_dict() for layer in self.layers]
        with open(path + ".json", "w") as file:
            json.dump({"layers": layer_data}, file)

        for i in range(len(self.layers)):
            self.layers[i].save_params(path=path + "_layer" + str(i))

    def load_params(self, folder_path, name):
        path = os.path.join(folder_path, name)

        with open(path + ".json", "r") as file:
            data = json.load(file)

        layers = data['layers']
        self.layers = []
        for layer in layers:
            match layer['type']:
                case "<class 'layer.fully_connected_layer.FullyConnectedLayer'>":
                    self.layers.append(FullyConnectedLayer(Shape(axis=layer['in_shape']), Shape(axis=layer['out_shape'])))
                case "<class 'activations.relu_layer.ActivationsReLULayer'>":
                    self.layers.append(ActivationsReLULayer())
                case "<class 'activations.sigmoid_layer.ActivationsSigmoidLayer'>":
                    self.layers.append(ActivationsSigmoidLayer())
                case "<class 'activations.softmax_layer.SoftmaxOutputLayer'>":
                    self.layers.append(SoftmaxOutputLayer())
                case "<class 'activations.tanh_layer.ActivationsTanhLayer'>":
                    self.layers.append(ActivationsTanhLayer())
                case "<class 'layer.flatten.Flatten'>":
                    self.layers.append(Flatten(Shape(axis=layer['in_shape']), Shape(axis=layer['out_shape'])))
                case "<class 'layer_2D.pooling_2D_layer.Pooling2DLayer'>":
                    self.layers.append(Pooling2DLayer(Shape(axis=layer['in_shape']), Shape(axis=layer['out_shape']),
                                                      kernel_size=Shape(axis=layer["kernel_size"]), pooling_type=PoolingType[layer["pooling_type"]],
                                                      stride=Shape(axis=layer["stride"])))
                case "<class 'layer_2D.convolution_layer.Conv2DLayer'>":
                    self.layers.append(Conv2DLayer(Shape(axis=layer['in_shape']), Shape(axis=layer['out_shape']),
                                                   Shape(axis=layer['kernel_size']), layer['num_filters'],
                                                   Shape(axis=layer['stride'])))
                case _:
                    print(layer['type'])
                    raise Exception("Unknown layer type")

        for i in range(len(self.layers)):
            if isinstance(self.layers[i], (FullyConnectedLayer, Conv2DLayer)):
                self.layers[i].load_params(path=path + "_layer" + str(i) + ".json")
