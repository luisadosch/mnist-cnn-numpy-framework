from layer.layer_interface import LayerInterface
from tensor import Shape, Tensor
from enum import Enum
import numpy as np
from numpy.lib.stride_tricks import as_strided

class PoolingType(Enum):
    MAX = "max"
    def __str__(self):
        return self.name.lower()


class Pooling2DLayer(LayerInterface):
    def __init__(self, in_shape: Shape, out_shape: Shape, kernel_size: Shape, pooling_type: PoolingType, stride: Shape,):  # in/out_shape: Höhe, Breite, Kanäle -> Filter entsprechen den Outputkanälen
        super().__init__()
        # check if out_shape is valid based on in_shape, kernel_size, and stride
        if (out_shape.axis[0] != (in_shape.axis[0] - kernel_size.axis[0]) / stride.axis[0] + 1
                or out_shape.axis[1] != (in_shape.axis[1] - kernel_size.axis[1]) / stride.axis[1] + 1):
            raise ValueError('out_shape must be (in_shape - kernel) / stride + 1)')

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.kernel_size = kernel_size
        self.pooling_type = pooling_type
        self.stride = stride
        self.mask = np.zeros(self.out_shape.axis, dtype=np.int8)


    def forward(self, in_tensors: list[Tensor], out_tensors: list[Tensor]):
        for in_tensor, out_tensor in zip(in_tensors, out_tensors):
            h_k, w_k = self.kernel_size.axis
            s_h, s_w = self.stride.axis
            H_out = (in_tensor.shape.axis[0] - h_k) // s_h + 1
            W_out = (in_tensor.shape.axis[1] - w_k) // s_w + 1

            s = in_tensor.elements.strides
            # shape: (H_out, W_out, h_k, w_k, C)
            shape = (H_out, W_out, h_k, w_k, in_tensor.shape.axis[2])
            strides = (s[0] * s_h, s[1] * s_w, s[0], s[1], s[2])
            windows = np.lib.stride_tricks.as_strided(in_tensor.elements, shape=shape, strides=strides)

            if self.pooling_type == PoolingType.MAX:
                # Max pooling über die (h_k, w_k)-Region
                out = np.max(windows, axis=(2, 3))

                # Optional: Maske für Backward Pass berechnen
                if hasattr(self, 'mask'):
                    # Fenster reshape für einfaches argmax
                    flat_windows = windows.reshape(H_out, W_out, h_k * w_k, -1)
                    max_indices = np.argmax(flat_windows, axis=2)
                    self.mask[:H_out, :W_out, :] = max_indices

                out_tensor.elements[:H_out, :W_out, :] = out

    def forward_old(self, in_tensors: list[Tensor], out_tensors: list[Tensor]):
        # Implement the forward pass for 2D pooling
        for in_tensor, out_tensor in zip(in_tensors, out_tensors):

            for i, pos_x in enumerate(range(0, in_tensor.shape.axis[0] - self.kernel_size.axis[0] + 1, self.stride.axis[0])):
                for j, pos_y in enumerate(range(0, in_tensor.shape.axis[1] - self.kernel_size.axis[1] + 1, self.stride.axis[1])):
                    
                    # Extract the window for pooling
                    window = in_tensor.elements[
                        pos_x:pos_x + self.kernel_size.axis[0], # Kernel height = height of the window
                        pos_y:pos_y + self.kernel_size.axis[1], # Kernel width = width of the window
                    ]
                    if self.pooling_type == PoolingType.MAX:
                        # Backpropagate for each channel
                        for c in range(window.shape[2]):
                            # Find the index of the max value in the window for this channel
                            index_number = np.argmax(window[:, :, c]) # argmax returns the index of the maximum value in the flattened array of the channel
                            self.mask[i, j, c] = index_number # store the index of the max value in the mask at position (i, j, c) - this is the position of the max value in the window
                            # Get the x and y coordinates of the max value in the window
                            dx, dy = np.unravel_index(index_number, window[:, :, c].shape) # unravel turns the index number into a tuple of (x, y) coordinates - because mask position is the number within the window
                            # Assign the max value to the output tensor
                            out_tensor.elements[i, j, c] = window[dx, dy, c] # assign the max value to the output tensor at position (i, j, c)

    def backward_old(self, out_tensors: list[Tensor], in_tensors: list[Tensor]):
        # Implement the backward pass for 2D max pooling
        for out_tensor, in_tensor in zip(out_tensors, in_tensors):
            # Reset the deltas of the input tensor
            in_tensor.deltas.fill(0)

            # Loop over output positions
            for i, pos_x in enumerate(range(0,
                                            in_tensor.shape.axis[0] - self.kernel_size.axis[0] + 1,
                                            self.stride.axis[0])):
                for j, pos_y in enumerate(range(0,
                                                in_tensor.shape.axis[1] - self.kernel_size.axis[1] + 1,
                                                self.stride.axis[1])):
                    
                    window = in_tensor.elements[
                        pos_x:pos_x + self.kernel_size.axis[0],
                        pos_y:pos_y + self.kernel_size.axis[1],
                    ]

                    if self.pooling_type == PoolingType.MAX:
                        # Backpropagate for each channel
                        for c in range(window.shape[2]):
                            # Find the index of the max value in the window for this channel
                            index_number = self.mask[i, j, c]  # get the index of the max value from the mask, index number is the position of max value in the window
                            dx, dy = np.unravel_index(index_number, window[:, :, c].shape) # get x and y coordinates of the max value in the window (because position is the number within the window)
                            # Route the gradient from the output back to this position
                            in_tensor.deltas[pos_x + dx, pos_y + dy, c] += out_tensor.deltas[i, j, c] # add the gradient from the output tensor to the input tensor at the position of the max value in the window

    def backward(self, out_tensors: list, in_tensors: list):
        for out_t, in_t in zip(out_tensors, in_tensors):
            in_t.deltas.fill(0)

            H_out, W_out, C = out_t.deltas.shape
            h_k, w_k = self.kernel_size.axis
            s_h, s_w = self.stride.axis

            # 1) grid of window origins
            i_idx = np.arange(H_out) * s_h
            j_idx = np.arange(W_out) * s_w
            base_x, base_y = np.meshgrid(i_idx, j_idx, indexing="ij")  # (H_out, W_out)

            # 2) unravel the flat mask → local offsets (dx, dy) shape (H_out, W_out, C)
            dx, dy = np.unravel_index(self.mask, (h_k, w_k))

            # 3) absolute positions in the input tensor
            #    base_x[...,None] has shape (H_out, W_out, 1)
            abs_x = base_x[..., None] + dx  # → (H_out, W_out, C)
            abs_y = base_y[..., None] + dy  # → (H_out, W_out, C)

            # 4) flatten everything in the same order
            flat_x = abs_x.ravel()  # length = H_out*W_out*C
            flat_y = abs_y.ravel()
            flat_g = out_t.deltas.ravel()

            # 5) build a matching channel-index array: 0,1,..,C-1 repeated for each (i,j)
            flat_c = np.tile(np.arange(C), H_out * W_out)

            # 6) scatter‑add in one go
            np.add.at(in_t.deltas, (flat_x, flat_y, flat_c), flat_g)

    def to_dict(self):
        return {
            "type": str(type(self)),
            "in_shape": self.in_shape.axis,
            "out_shape": self.out_shape.axis,
            "kernel_size": self.kernel_size.axis,
            "stride": self.stride.axis,
            "pooling_type": self.pooling_type.name
        }

    @classmethod
    def from_dict(cls, data):
        layer = cls(
            in_shape=Shape(axis=data["in_shape"]),
            out_shape=Shape(axis=data["out_shape"]),
            kernel_size=Shape(axis=data["kernel_size"]),
            stride=Shape(axis=data["stride"]),
            pooling_type=PoolingType[data["pooling_type"]]
        )
        return layer
