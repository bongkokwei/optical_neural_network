import torch.nn as nn
from functools import partial

from ..layer.optical_layer import OpticalLinearLayer
from ..layer.adaptive_optical_layer import AdaptiveOpticalLayer
from ..layer.segmented_optical_layer import SegmentedOpticalLayer
from .utils import create_optical_layers, create_reduction_layers


class OpticalMNISTClassifier(nn.Module):
    def __init__(
        self,
        input_size=784,
        num_classes=10,
        num_layers=3,
        num_optical_input=32,
        num_optical_layers=2,
        device_max_inputs=32,
        dropout_rate=0.2,
    ):
        super(OpticalMNISTClassifier, self).__init__()

        self.num_layers = num_layers
        if self.num_layers > 0:
            self.reduction_layers = create_reduction_layers(
                input_size,
                num_optical_input,
                num_layers,
            )

        self.dropout = nn.Dropout(dropout_rate)

        itf_layer = partial(OpticalLinearLayer)

        # Optical layers
        self.optical_layers = create_optical_layers(
            num_layers=num_optical_layers,
            initial_size=num_optical_input,
            final_size=num_classes,
            device_max_inputs=device_max_inputs,  # size of the physical gmzi
            optical_layer=itf_layer,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)

        # Reduce dimensionality
        if self.num_layers > 0:
            x = self.reduction_layers(x)
            x = self.dropout(x)
        # Optical layers
        x = self.optical_layers(x)

        return x
