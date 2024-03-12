
from shared import DEVICE, VALIDATION
from quantization import quantize_model_per_layer, dequantize_weights_per_layer, reinject_simulated_quantized_weights
from evaluate import evaluate_model
from infer import load_model


def main_quantization(exp: int = 201, device=DEVICE, num_bits: int = 16, eval_flag: bool = True):
    original_model, dl_dict, model_config = load_model(exp, device=device, batch_size=16)
    quantized_model, _, _ = load_model(exp, device=device, get_data_loaders_flag=False)
    print(f"model {exp} has  {original_model.count_parameters()} parameters")
    quantized_weights, quantized_params = quantize_model_per_layer(original_model, num_bits=num_bits)
    params_dequant = dequantize_weights_per_layer(quantized_weights, quantized_params)
    reinject_simulated_quantized_weights(quantized_model, params_dequant, device=device)
    if eval_flag:
        evaluate_model(quantized_model, dl_dict, phase=VALIDATION)
        print("Original model")
        evaluate_model(original_model, dl_dict, phase=VALIDATION)
    return quantized_weights, quantized_params


if __name__ == "__main__":
    main_quantization(num_bits=8)
