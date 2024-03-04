import torch
from shared import DEVICE
import numpy as np


def get_array_size_in_bytes(x, print_flag: bool = True) -> int:
    size_in_bytes = (x.size * x.itemsize)
    if print_flag:
        print(f"{size_in_bytes/1024/1024:.2f} Mb")
    return size_in_bytes


def quantize_model_per_layer(model: torch.nn.Module,  num_bits: int = 16, device: str = DEVICE) -> dict:
    layer_weights = model.named_parameters()
    quantized_weights = {}

    for name, weights in layer_weights:
        if 'weight' in name:
            quantized_weights[name] = quantize_weights(weights.detach().cpu().numpy(), num_bits)
    compressed_weights_size = get_array_size_in_bytes(
        np.concatenate([el.flatten() for _, el in quantized_weights.items()]))
    params = torch.cat([p.flatten() for p in model.parameters() if p.requires_grad]).detach().cpu().numpy()
    original_weights_size = get_array_size_in_bytes(params)
    ratio = original_weights_size/compressed_weights_size
    print(f"compression ratio = {ratio:0.3f}")
    return quantized_weights


def quantize_weights(weights: np.ndarray, num_bits: int = 16) -> np.ndarray:
    assert num_bits in [16, 8], "Number of bits must be 16 or 8"
    # Compute range of the weights
    weight_range = np.abs(weights).max()
    # Compute  scale factor for quantization store on SIGNED INTS of size num_bits
    # -2^(num_bits-1) to 2^(num_bits-1)-1
    scale_factor = 2**(num_bits-1)
    # Quantize the weights
    quantized_weights = np.round(weights / weight_range * scale_factor)
    quantized_weights = quantized_weights.clip(-scale_factor, scale_factor-1)

    # Convert to signed int for storage
    if num_bits == 16:
        int_type = np.int16
    elif num_bits == 8:
        int_type = np.int8
    quantized_weights = quantized_weights.astype(int_type)
    # print(f"{np.std(quantized_weights)}")
    return quantized_weights
