import gzip

# import jax
import io
import json
import os
import pickle
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def write_to_json(filename: Path, content: dict):
    """Write data to a JSON file.

    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.suffix == ".json"
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(content, file)


def load_image_file_to_torch(image_filename, scale_factor: float = 1.0):
    rgb = load_numpy_image(image_filename, scale_factor=scale_factor)
    rgb = numpy_to_torch_image(rgb)
    return rgb


def numpy_to_torch_image(numpy_image):
    image = torch.from_numpy(numpy_image.astype("float32") / 255.0)
    image = image[:, :, :3]

    image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    return image


def load_numpy_image(image_filename, scale_factor: float = 1.0):
    pil_image = Image.open(image_filename)
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
    image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
    if len(image.shape) == 2:
        image = image[:, :, None].repeat(3, axis=2)
    assert len(image.shape) == 3
    assert image.dtype == np.uint8
    assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
    return image


# load everything onto cpu
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        # elif module == "jax.interpreters.xla" and name == "DeviceArray":
        #     return lambda b: jax.device_put(io.BytesIO(b), jax.devices("cpu")[0])
        else:
            return super().find_class(module, name)


def load_gzip_file(file_name):
    with gzip.open(file_name, "rb") as f:
        traj = CPU_Unpickler(f).load()
    return traj


def save_gzip_file(data, file_name):
    assert file_name[-3:] == "pkl"
    with gzip.open(file_name, "wb") as f:
        pickle.dump(data, f, protocol=4)


def generate_random_uuid():
    return str(uuid.uuid4().hex)


def create_folder(_dir, remove_exists=False):
    if os.path.exists(_dir) and remove_exists:
        print(f"Removing existing directory {_dir}")
        shutil.rmtree(_dir, ignore_errors=True)
    os.makedirs(_dir, exist_ok=True)


def load_optical_flow(optical_flow_filename):
    optical_flow = np.load(optical_flow_filename)["flow"]
    return optical_flow


def load_tracking_data(tracking_data_filename):
    tracking_data = np.load(tracking_data_filename)
    return tracking_data


def save_svg(fig, path: Path) -> None:
    import svg

    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("w") as f:
        # This hack makes embedded images work.
        f.write(
            str(fig)
            .replace("href", "xlink:href")
            .replace("<svg", '<svg xmlns:xlink="http://www.w3.org/1999/xlink"')
        )
    actual_width = float(
        subprocess.check_output(f"inkscape -D {path} --query-width".split(" "))
        .decode()
        .strip()
    )
    print(
        "When importing this SVG figure, make sure to multiply the width by "
        f"{actual_width / fig.width}"
    )


def combine_roots(cfg, dataparser_outputs, downscale_factor):

    print("Combining data with other roots", cfg.other_roots)

    total_dataparser_outputs = [dataparser_outputs]
    for other_root in cfg.other_roots:
        other_dataparser = DNeRFDataParserConfig(
            data=Path(other_root),
            center_method="focus",
            downscale_factor=downscale_factor,
        ).setup()
        total_dataparser_outputs.append(
            other_dataparser.get_dataparser_outputs(split="train")
        )

    # check joint positions and pad to the same length with zeros
    max_len = -1
    for i, other_dataparser_outputs in enumerate(total_dataparser_outputs):
        joint_positions = other_dataparser_outputs.metadata["joint_positions"]
        max_len = max(max_len, len(list(joint_positions.values())[0]))

    for i, other_dataparser_outputs in enumerate(total_dataparser_outputs):
        joint_positions = other_dataparser_outputs.metadata["joint_positions"]
        for key, value in joint_positions.items():
            if len(value) < max_len:
                pad_len = max_len - len(value)
                pad_value = torch.zeros(
                    (pad_len,), dtype=value.dtype, device=value.device
                )
                joint_positions[key] = torch.cat([value, pad_value], dim=0)

        other_dataparser_outputs.metadata["joint_positions"] = joint_positions

    return total_dataparser_outputs