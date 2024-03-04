import torch
from shared import ROOT_DIR, OUTPUT_FOLDER_NAME, DEVICE, NAME, VALIDATION, TRAIN
from experiments import get_experiment_config, get_training_content


def quantize_model(model: torch.nn.Module,  device: str = DEVICE):
    pass


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


def main_quantization(exp: int = 201, device=DEVICE):
    model = load_model(exp, device=device)
    quantize_model(model, device=device)


if __name__ == "__main__":
    main_quantization()
