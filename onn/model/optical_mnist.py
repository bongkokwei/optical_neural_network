import torch.nn as nn

from ..layer.optical_layer import (
    OpticalLinearLayer,
    SegmentedOpticalLayer,
)

from .utils import create_optical_layers, create_reduction_layers


class OpticalMNISTClassifier(nn.Module):
    def __init__(
        self,
        input_size=784,
        num_classes=10,
        num_layers=3,
        num_optical_input=32,
        num_optical_layers=2,
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

        itf_layer = lambda in_features, out_features, num_modes: OpticalLinearLayer(
            in_features=in_features,
            out_features=out_features,
            num_modes=num_modes,
        )

        # Optical layers
        self.optical_layers = create_optical_layers(
            num_layers=num_optical_layers,
            initial_size=num_optical_input,
            final_size=num_classes,
            num_modes=None,  # size of the physical gmzi
            optical_layer=itf_layer,
        )

        nn.Dropout(dropout_rate)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)

        # Reduce dimensionality
        if self.num_layers > 0:
            x = self.reduction_layers(x)

        # Optical layers
        x = self.optical_layers(x)

        return x


class AllOpticalMNISTClassifier(nn.Module):
    def __init__(
        self,
        num_optical_layers=3,
        input_size=784,
        num_classes=10,
        device_max_inputs=16,
    ):
        super().__init__()

        itf_layer = lambda in_features, out_features, num_modes: SegmentedOpticalLayer(
            in_features=in_features,
            out_features=out_features,
            num_modes=num_modes,
            device_max_inputs=device_max_inputs,
        )

        # Optical layers
        self.optical_layers = create_optical_layers(
            num_layers=num_optical_layers,
            initial_size=input_size,
            final_size=num_classes,
            num_modes=None,  # size of the physical gmzi
            optical_layer=itf_layer,
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the input
        x = self.optical_layers(x)

        return x
