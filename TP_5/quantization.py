import torch
from shared import DEVICE
import numpy as np
from typing import Tuple


def get_array_size_in_bytes(x: np.ndarray, print_flag: bool = True) -> int:
    size_in_bytes = (x.size * x.itemsize)
    if print_flag:
        print(f"{size_in_bytes/1024/1024:.2f} Mb")
    return size_in_bytes


def quantize_model_per_layer(
    model: torch.nn.Module,
    num_bits: int = 16,
) -> Tuple[dict, dict]:
    layer_weights = model.named_parameters()
    quantized_weights = {}
    quantization_parameters = {}
    for name, weights in layer_weights:
        if 'weight' in name:
            quantized_weights[name], quantization_parameters[name] = quantize_weights(
                weights.detach().cpu().numpy(), num_bits)
    compressed_weights_size = get_array_size_in_bytes(
        np.concatenate([el.flatten() for _, el in quantized_weights.items()]))
    params = torch.cat([p.flatten() for p in model.parameters() if p.requires_grad]).detach().cpu().numpy()
    original_weights_size = get_array_size_in_bytes(params)
    ratio = original_weights_size/compressed_weights_size
    print(f"compression ratio = {ratio:0.3f}")

    return quantized_weights, quantization_parameters


def dequantize_weights_per_layer(
    quantized_weights: dict,
    quantization_parameters: dict
) -> dict:
    decompressed_weights = {}
    for name in quantized_weights.keys():
        qweights = quantized_weights[name]
        qparams = quantization_parameters[name]
        decompressed_weights[name] = dequantize_weights(qweights, **qparams)
    return decompressed_weights


def quantize_weights(weights_uncentered: np.ndarray, num_bits: int = 16) -> Tuple[np.ndarray, dict]:
    zero_point = 0.  # could be the mean of the weights
    weights = weights_uncentered - zero_point
    if num_bits not in [16, 8]:
        print("Nonstandard: Number of bits better be 16 or 8 - otherwise need to pack but not done here!")
    # Compute range of the weights
    weight_range = np.abs(weights).max()
    # Compute  scale factor for quantization store on SIGNED INTS of size num_bits
    # -2^(num_bits-1) to 2^(num_bits-1)-1
    scale_factor = 2**(num_bits-1)
    # Quantize the weights
    quantized_weights = np.round(weights / weight_range * scale_factor)
    quantized_weights = quantized_weights.clip(-scale_factor, scale_factor-1)

    # Convert to signed int for storage
    if num_bits <= 16:
        int_type = np.int16
    if num_bits <= 8:
        int_type = np.int8
    quantized_weights = quantized_weights.astype(int_type)
    # print(f"{np.std(quantized_weights)}")
    quantization_parameters = {"scale": weight_range, "zero_point": zero_point, "num_bits": num_bits}
    return quantized_weights, quantization_parameters


def dequantize_weights(
    quantized_weights: np.ndarray,
    scale: float = 1.,
    zero_point: float = 0.,
    num_bits: int = 16
) -> np.ndarray:
    bit_dynamic = 2.**(num_bits-1)
    return zero_point + quantized_weights.astype(np.float32) * scale/bit_dynamic
