import numpy as np
from layer.layer_interface import LayerInterface
from tensor import Shape, Tensor
import json


class Conv2DLayer(LayerInterface):

    # in_shape = Shape((4, 3, 2)), out_shape = Shape((3, 2, 2)), kernel_size = Shape((2, 2)), num_filters = 2
    def __init__(self, in_shape: Shape, out_shape: Shape, kernel_size: Shape, num_filters: int, stride=Shape([1, 1])):  # in/out_shape: Höhe, Breite, Kanäle -> Filter entsprechen den Outputkanälen
        super().__init__()
        if (out_shape.axis[0] != (in_shape.axis[0] - kernel_size.axis[0]) / stride.axis[0] + 1
                or out_shape.axis[1] != (in_shape.axis[1] - kernel_size.axis[1]) / stride.axis[1] + 1):
            raise ValueError('out_shape must be (in_shape - kernel) / stride + 1)')

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.stride = stride

        fan_in = kernel_size.axis[0] * kernel_size.axis[1] * in_shape.axis[2]
        self.weights = Tensor(elements=np.random.randn(
            kernel_size.axis[0], kernel_size.axis[1], in_shape.axis[2], num_filters
        ) * np.sqrt(2. / fan_in))
        self.bias = Tensor(elements=np.zeros(num_filters))
        """        
        self.weights = Tensor(elements=np.random.uniform(low=-0.5, high=0.5, size=(
            kernel_size.axis[0], kernel_size.axis[1], in_shape.axis[2], num_filters)))
        self.bias = Tensor(elements=np.random.uniform(low=-0.5, high=0.5, size=num_filters))
        """

    def forward(self, in_tensors: list, out_tensors: list):
        for in_tensor, out_tensor in zip(in_tensors, out_tensors):
            out_height = (in_tensor.shape.axis[0] - self.weights.shape.axis[0]) // self.stride.axis[0] + 1
            out_width = (in_tensor.shape.axis[1] - self.weights.shape.axis[1]) // self.stride.axis[1] + 1

            s = in_tensor.elements.strides
            shape = (out_height, out_width, self.weights.shape.axis[0], self.weights.shape.axis[1], self.weights.shape.axis[2])
            strides = (s[0] * self.stride.axis[0], s[1] * self.stride.axis[1], s[0], s[1], s[2])
            x_windows = np.lib.stride_tricks.as_strided(in_tensor.elements, shape=shape, strides=strides)

            out = np.einsum('xyhwc,hwco->xyo', x_windows, self.weights.elements)

            out += self.bias.elements
            out_tensor.elements = out


    def forward_old(self, in_tensors: list[Tensor], out_tensors: list[Tensor]):
        for in_tensor, out_tensor in zip(in_tensors, out_tensors):
            for pos_x in range(0, in_tensor.shape.axis[0]-self.kernel_size.axis[0]+1, self.stride.axis[0]):  # Durchlauf durchs Bild in X-Richtung
                for pos_y in range(0, in_tensor.shape.axis[1]-self.kernel_size.axis[1]+1, self.stride.axis[1]):  # Durchlauf durchs Bild in Y-Richtung
                    for f in range(self.num_filters):   # out_channels / Filter
                        out_tensor.elements[int(pos_x / self.stride.axis[0])][int(pos_y / self.stride.axis[1])][f] = self.bias.elements[f]
                        for k in range(in_tensor.shape.axis[2]):    # in_channels
                            for i in range(self.kernel_size.axis[0]):   # Durchlauf in i des Filters + Bild
                                for j in range(self.kernel_size.axis[1]):   # Durchlauf in j des Filters + Bild
                                    out_tensor.elements[int(pos_x/self.stride.axis[0])][int(pos_y/self.stride.axis[1])][f] \
                                        += in_tensor.elements[pos_x+i][pos_y+j][k] * self.weights.elements[i][j][k][f]
        """
        Forward Convolution:

        1. Filter durchlaufen und für jeden Filter Convolution auf jeden Kanal anwenden.
        2. Ergebnis Summe aller Kanäle in Filter 1 ist Output1 und dasselbe für die anderen
        3. Auf jeden output den jeweiligen Bias
        """

    def backward(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        # swap input/output channels for backward
        weights_transposed = self.weights.elements.transpose((0, 1, 3, 2))
        # Rotate weights by 180
        weights_rotated = np.rot90(weights_transposed, 2, axes=(0, 1))

        kH, kW = self.kernel_size.axis[0], self.kernel_size.axis[1]

        for in_tensor, out_tensor in zip(in_tensors, out_tensors):
            d_out = out_tensor.deltas
            in_H, in_W, _ = in_tensor.shape.axis
            _, _, out_ch = out_tensor.shape.axis

            pad_h = kH - 1
            pad_w = kW - 1
            d_out_padded = np.pad(d_out,
                                  pad_width=((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                                  mode='constant')

            # 2. Create sliding-window view of padded deltas instead of loops through the filters with entire data
            s = d_out_padded.strides
            view_shape = (in_H, in_W, kH, kW, out_ch)
            view_strides = (s[0], s[1], s[0], s[1], s[2])
            d_out_view = np.lib.stride_tricks.as_strided(d_out_padded,
                                                         shape=view_shape,
                                                         strides=view_strides,
                                                         writeable=False)

            # 3. Perform convolution using Einstein summation
            in_deltas = np.einsum('xyijf,ijfk->xyk', d_out_view, weights_rotated)
            in_tensor.deltas += in_deltas

    def backward_old(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        weights_transposed = self.weights.elements.transpose((0, 1, 3, 2))  # [kH, kW, out_ch, in_ch]
        weights_rotated = np.rot90(weights_transposed, 2, axes=(0, 1))  # 180° flip
        for in_tensor, out_tensor in zip(in_tensors, out_tensors):
            for out_x in range(out_tensor.shape.axis[0]):
                for out_y in range(out_tensor.shape.axis[1]):
                    for f in range(self.num_filters):  # output channel
                        for k in range(in_tensor.shape.axis[2]):  # input channel
                            for i in range(self.kernel_size.axis[0]):
                                for j in range(self.kernel_size.axis[1]):
                                    # berechnet, welche stellen der in_tensoren abhängig von den out_tensoren betroffen sind:
                                    in_x = out_x + self.kernel_size.axis[0] - 1 - i
                                    in_y = out_y + self.kernel_size.axis[1] - 1 - j
                                    if 0 <= in_x < in_tensor.shape.axis[0] and 0 <= in_y < in_tensor.shape.axis[1]:
                                        in_tensor.deltas[in_x][in_y][k] += out_tensor.deltas[out_x][out_y][f] * weights_rotated[i][j][f][k]


    def calculate_delta_weights(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        kH, kW, in_ch, out_ch = self.weights.shape.axis
        stride_h, stride_w = self.stride.axis

        self.weights.deltas.fill(0)
        self.bias.deltas.fill(0)

        for in_tensor, out_tensor in zip(in_tensors, out_tensors):
            in_H, in_W, _ = in_tensor.shape.axis
            out_H, out_W, _ = out_tensor.shape.axis

            # Sliding-Window auf das Eingabebild
            s = in_tensor.elements.strides
            shape = (out_H, out_W, kH, kW, in_ch)
            strides = (s[0] * stride_h, s[1] * stride_w, s[0], s[1], s[2])
            x_windows = np.lib.stride_tricks.as_strided(in_tensor.elements, shape=shape, strides=strides)

            # einsum: berechnet aus x_windows: 'xyhwc' und out_tensor.deltas: 'xyo' -> self.weights.deltas: 'hwco'
            self.weights.deltas += np.einsum('xyhwc,xyo->hwco', x_windows, out_tensor.deltas)

            # Bias-Gradient: Summe über alle Positionen im Output für jeden Filter
            self.bias.deltas += np.sum(out_tensor.deltas, axis=(0, 1))

    def calculate_delta_weights_old(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        for in_tensor, out_tensor in zip(in_tensors, out_tensors):
            for out_x in range(out_tensor.shape.axis[0]):
                for out_y in range(out_tensor.shape.axis[1]):
                    for f in range(self.num_filters):   # output channel
                        self.bias.deltas[f] += out_tensor.deltas[out_x][out_y][f]
                        for k in range(in_tensor.shape.axis[2]):    # input channel
                            for i in range(self.kernel_size.axis[0]):
                                for j in range(self.kernel_size.axis[1]):
                                    in_x = out_x + i
                                    in_y = out_y + j
                                    self.weights.deltas[i][j][k][f] += in_tensor.elements[in_x][in_y][k] * out_tensor.deltas[out_x][out_y][f]

    def save_params(self, path):
        data = {
            "weights": self.weights.elements.tolist(),
            "bias": self.bias.elements.tolist(),
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
            "out_shape": self.out_shape.axis,
            "kernel_size": self.kernel_size.axis,
            "num_filters": self.num_filters,
            "stride": self.stride.axis,
        }

    @classmethod
    def from_dict(cls, data):
        layer = cls(
            in_shape=Shape(axis=data["in_shape"]),
            out_shape=Shape(axis=data["out_shape"]),
            kernel_size=data["kernel_size"],
            num_filters=data["num_filters"],
            stride=Shape(axis=data["stride"]),
        )
        return layer
