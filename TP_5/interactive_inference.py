from interactive_pipe import interactive_pipeline, interactive, KeyboardControl
from interactive_pipe.headless.pipeline import HeadlessPipeline
from interactive_pipe.graphical.qt_gui import InteractivePipeQT
from interactive_pipe.graphical.mpl_gui import InteractivePipeMatplotlib
from interactive_pipe.data_objects.image import Image
from batch_processing import Batch
import argparse
from shared import DEVICE, ROOT_DIR, OUTPUT_FOLDER_NAME, NAME, N_PARAMS
from typing import List
from experiments import get_experiment_config, get_training_content
import torch
import sys
from pathlib import Path
from data_loader import load_npy_files
import numpy as np
from augmentations import augment_wrap_roll, augment_flip


def parse_command_line(parser: Batch = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description='Segmentation inference',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    iparse = parser.add_argument_group("Segmentation inference")
    iparse.add_argument("-e",  "--experiments", type=int, nargs="+", required=True,
                        help="Experiment ids to be inferred sequentially")
    iparse.add_argument("-m", "--model-root", type=str, default=ROOT_DIR/OUTPUT_FOLDER_NAME)
    iparse.add_argument("-d", "--device", type=str, default=DEVICE,
                        choices=["cpu", "cuda"] if DEVICE == "cuda" else ["cpu"])
    iparse.add_argument("-gui", "--gui", type=str, default="qt", choices=["qt", "mpl"])
    iparse.add_argument("--preload", action="store_true", help="Preload npy files")
    return parser


def npy_loading_batch(input: Path, args: argparse.Namespace) -> dict:
    """Wrapper to load npy files from a directory using batch_processing
    """
    label_path = input.parent.parent/"labels"/input.name
    label_buffer = None
    if label_path.exists():
        if args.preload:
            label_buffer = load_npy_files(label_path)

        # label_path = label_path
    else:
        label_path = None
        label_buffer = None

    if args.preload:
        return {"name": input.name, "path": input, "buffer": load_npy_files(input), "label_path": label_path, "label_buffer": label_buffer}
    else:
        return {"name": input.name, "path": input, "buffer": None, "label_path": label_path, "label_buffer": None}


@interactive(
    idx=KeyboardControl(value_default=0, value_range=[0, 10000], modulo=True, keyup="right", keydown="left"),
)
def selector(img_list: List[torch.Tensor], global_params: dict = {}, idx: int = 0):
    valid_idx = idx % len(img_list)
    global_params["idx"] = valid_idx
    img = img_list[valid_idx]["buffer"]
    label_img = img_list[valid_idx]["label_buffer"]
    if img is None:
        img = load_npy_files(img_list[valid_idx]["path"])
    if label_img is None and img_list[valid_idx]["label_path"] is not None:
        label_img = load_npy_files(img_list[valid_idx]["label_path"])
    if label_img is not None:
        label_img = label_img
    title = f"image={global_params.get('idx', 0)}"
    title += f"\n{img_list[valid_idx]['name']}"
    global_params["__output_styles"]["img"] = {"title": title}
    return img, label_img


def resize_images(img: torch.Tensor):
    img = img.view(1, 1, img.shape[-2], img.shape[-1])
    img = torch.nn.functional.interpolate(img, scale_factor=8., mode="nearest")
    img = img.squeeze(0).squeeze(0)
    return img


@interactive(
    adapt_dynamic_range=(True,),
)
def display_tensor(img: torch.Tensor, adapt_dynamic_range=True, global_params: dict = {}):
    if adapt_dynamic_range:
        mini, maxi = torch.min(img), torch.max(img)
    else:
        mini, maxi = -0.2, 0.2
    img_rescaled = (img - mini)/(maxi-mini)
    img_rescaled = resize_images(img_rescaled)
    img_rescaled = img_rescaled.unsqueeze(-1).repeat(1, 1, 3)
    int_array = (img_rescaled.cpu().numpy())
    return int_array


def display_mask(mask, global_params: dict = {}):
    if mask is None:
        return None
    mask = mask.float()
    mask_resize = resize_images(mask)
    # mask_resize = mask
    return mask_resize.squeeze(0).cpu().numpy().astype(np.float32)


@interactive(
    model_index=KeyboardControl(value_default=0, value_range=[
                                0, 10000], modulo=True, keyup="pageup", keydown="pagedown"),
)
def model_selector(model_dic: dict, global_params: dict = {}, model_index: int = 0):
    model_selections = list(model_dic.keys())
    selected_model_index = model_index % len(model_selections)
    model_name = model_selections[selected_model_index]
    global_params["model_name"] = model_name
    print(model_name, model_dic[model_name]["model"][NAME])
    model_pretty_name = model_dic[model_name]["model"][NAME]
    n_params = model_dic[model_name]["model"][N_PARAMS]
    global_params["__output_styles"]["infered_mask"] = {
        "title": f"Inference\n{model_pretty_name}\n{n_params/1000:.1f}k params",
    }
    return model_dic[model_name]["torch_model"]


def inference(img: torch.Tensor, model: torch.nn.Module, global_params: dict = {}):
    with torch.no_grad():
        output = model(img.to(DEVICE).unsqueeze(0))
    predicted_mask = (torch.sigmoid(output[0, 0, ...]) > 0.5)
    return predicted_mask


@interactive(
    shift=(0, [0, 36]),
    noise=(0., [0, 0.1]),
)
def modify(img: torch.Tensor, label_image: torch.Tensor, shift=0, noise=0., global_params: dict = {}):
    img, label_image = augment_wrap_roll(img, label_image, shift=shift)
    img = img + noise*torch.randn_like(img)
    return img, label_image


def segmentation_demo(img_list: List[torch.Tensor], model_dict: dict):
    img, label_image = selector(img_list)
    model = model_selector(model_dict)
    img, label_image = modify(img, label_image)
    infered_mask = inference(img, model)
    img = display_tensor(img)
    label_image = display_mask(label_image)
    infered_mask = display_mask(infered_mask)
    # .float().cpu().numpy()
    # infered_mask = display_tensor(infered_mask)
    return img, infered_mask, label_image


def main(argv: List[str]):
    """Paired signals and noise in folders"""
    batch = Batch(argv)
    batch.set_io_description(
        input_help='input image files',
        output_help=argparse.SUPPRESS
    )
    parser = parse_command_line()
    args = batch.parse_args(parser)
    device = args.device
    batch.set_multiprocessing_enabled(False)
    parser = parse_command_line()
    img_list = batch.run(npy_loading_batch)
    # print(len(img_list))
    # print(img_list[0].shape)
    model_list = {}
    for exp in args.experiments:
        config = get_experiment_config(exp)
        model, _, dl_dict = get_training_content(config, device=DEVICE)
        model.load_state_dict(torch.load(Path(args.model_root)/config[NAME]/"best_model.pt"))
        model.eval()
        model.to(device)
        model_list[config[NAME]] = {
            "torch_model": model,
            **config
        }
    pip = HeadlessPipeline.from_function(segmentation_demo, cache=False)
    gui = args.gui
    print(f">>>>>>>>>>>>>>>>>>>> Using {device}")
    if gui == "qt":
        app = InteractivePipeQT(
            pipeline=pip,
            name="Corrosion segmentation",
            # size=(1000, 1000)
        )
    else:
        app = InteractivePipeMatplotlib(
            pipeline=pip,
            name="Well corrosion segmentation",
            size=None
        )
    app(img_list, model_list)


if __name__ == "__main__":
    main(sys.argv[1:])
