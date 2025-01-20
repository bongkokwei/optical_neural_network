import torch.nn as nn

from ..layer.optical_layer import OpticalLinearLayer
from ..layer.adaptive_optical_layer import AdaptiveOpticalLayer
from ..layer.segmented_optical_layer import SegmentedOpticalLayer
from .utils import create_optical_layers, create_reduction_layers


class OpticalCancerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        device_max_inputs=16,
        num_optical_layers=3,
    ):
        super(OpticalCancerClassifier, self).__init__()

        itf_layer = (
            lambda in_features, out_features, device_max_inputs: AdaptiveOpticalLayer(
                in_features=in_features,
                out_features=out_features,
                device_max_inputs=device_max_inputs,
            )
        )

        # Optical layers
        self.optical_layers = create_optical_layers(
            num_layers=num_optical_layers,
            initial_size=input_dim,
            final_size=1,
            device_max_inputs=device_max_inputs,  # size of the physical gmzi
            optical_layer=itf_layer,
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the input
        x = self.optical_layers(x)

        return x.squeeze()
