
import torch
from shared import ROOT_DIR, OUTPUT_FOLDER_NAME, DEVICE, NAME, VALIDATION
from experiments import get_experiment_config, get_training_content
from quantization import quantize_model_per_layer, dequantize_weights_per_layer, reinject_simulated_quantized_weights
from evaluate import evaluate_model


def load_model(exp: int, device: str = DEVICE, get_data_loaders_flag: bool = True):
    output_dir = ROOT_DIR/OUTPUT_FOLDER_NAME
    config = get_experiment_config(exp)
    inference_dir = output_dir/(config[NAME]+"_inference")
    inference_dir.mkdir(exist_ok=True, parents=True)
    model, _, dl_dict = get_training_content(config, device=device, get_data_loaders_flag=get_data_loaders_flag)
    model.load_state_dict(torch.load(output_dir/config[NAME]/"best_model.pt"))
    model.eval()
    model.to(device)
    return model, dl_dict


def main_quantization(exp: int = 201, device=DEVICE, num_bits: int = 16, eval_flag: bool = True):
    original_model, dl_dict = load_model(exp, device=device)
    quantized_model, _ = load_model(exp, device=device, get_data_loaders_flag=False)
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
