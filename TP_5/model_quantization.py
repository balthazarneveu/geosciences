
import torch
from shared import ROOT_DIR, OUTPUT_FOLDER_NAME, DEVICE, NAME
from experiments import get_experiment_config, get_training_content
from quantization import quantize_model_per_layer


def load_model(exp: int, device: str = DEVICE):
    output_dir = ROOT_DIR/OUTPUT_FOLDER_NAME
    config = get_experiment_config(exp)
    inference_dir = output_dir/(config[NAME]+"_inference")
    inference_dir.mkdir(exist_ok=True, parents=True)
    model, _, dl_dict = get_training_content(config, device=device)
    model.load_state_dict(torch.load(output_dir/config[NAME]/"best_model.pt"))
    model.eval()
    model.to(device)
    return model


def main_quantization(exp: int = 201, device=DEVICE, num_bits: int = 16):
    model = load_model(exp, device=device)
    quantized_weights = quantize_model_per_layer(model, num_bits=num_bits, device=device)
    return quantized_weights


if __name__ == "__main__":
    main_quantization()
