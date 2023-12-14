# Adapted from https://github.com/icaros-usc/dqd/blob/main/experiments/lsi_clip/make_archive_collage.py

import matplotlib
import matplotlib.font_manager

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = "Palatino"
matplotlib.rc("font", size=20)

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from torchvision.utils import make_grid, save_image


def make_archive_collage(archive_filename, prompt, save_all=False):
    # min and max index for rows then columns (row major).
    # The archive is shape (200, 200) indexed from [0, 200).
    archive_index_range = ((0, 20), (0, 20))
    picture_frequency = (4, 4)

    # Save all grid images separately.
    gen_output_dir = os.path.join("logs/vis", archive_filename.split("/")[1])
    gen_output_dir = Path(gen_output_dir)
    if not gen_output_dir.is_dir():
        gen_output_dir.mkdir()

    model_id = "stabilityai/stable-diffusion-2-1-base"

    torch_dtype = torch.float16
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,  # for faster inference
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    # Read the archive from the log (pickle file)
    df = pd.read_pickle(archive_filename)

    imgs = []
    for j in reversed(range(picture_frequency[1])):
        for i in range(picture_frequency[0]):
            delta_i = archive_index_range[0][1] - archive_index_range[0][0]
            delta_j = archive_index_range[1][1] - archive_index_range[1][0]
            index_i_lower = int(
                delta_i * i / picture_frequency[0] + archive_index_range[0][0]
            )
            index_i_upper = int(
                delta_i * (i + 1) / picture_frequency[0] + archive_index_range[0][0]
            )
            index_j_lower = int(
                delta_j * j / picture_frequency[1] + archive_index_range[1][0]
            )
            index_j_upper = int(
                delta_j * (j + 1) / picture_frequency[1] + archive_index_range[1][0]
            )
            print(i, j, index_i_lower, index_i_upper, index_j_lower, index_j_upper)

            query_string = f"{index_i_lower} <= index_0 & index_0 < {index_i_upper} &"
            query_string += f"{index_j_lower} <= index_1 & index_1 < {index_j_upper}"
            print(query_string)
            df_cell = df.query(query_string)

            if not df_cell.empty:
                if save_all:
                    for _ in range(len(df_cell)):
                        sol = df_cell.iloc[_]
                        latents = torch.from_numpy(
                            sol[5:-1].values.astype(np.float16)
                        ).to(device)
                        latents = latents.reshape((1, 4, 64, 64))

                        image = pipe(
                            prompt,
                            num_images_per_prompt=1,
                            latents=latents,
                            # num_inference_steps=1,  # for test
                        ).images[0]

                        img = torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255.0

                        fp = os.path.join(gen_output_dir, f"{i}_{j}_{_}.png")
                        save_image(img, fp)

                # sol = df_cell.iloc[df_cell["objective"].argmax()]
                sol = df_cell.sample(n=1).iloc[0]

                latents = torch.from_numpy(sol[5:-1].values.astype(np.float16))
                latents = latents.reshape((1, 4, 64, 64)).to(device)

                image = pipe(
                    prompt,
                    num_images_per_prompt=1,
                    latents=latents,
                    # num_inference_steps=1,  # for test
                ).images[0]

                img = torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255.0
            else:
                img = torch.zeros((3, 512, 512))

            imgs.append(img)

    img_grid = make_grid(imgs, nrow=picture_frequency[0], padding=8, pad_value=1.0)

    fp = os.path.join(gen_output_dir, f"collage.png")
    save_image(img_grid, fp)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "a photo of an astronaut riding a horse on mars"

    # Note that only final archives encode latent codes.
    archive_filename = (
        f"logs/map_elites_{prompt}/qdhf(n=10000)|online|fixed_archive_00000200.pkl"
    )
    make_archive_collage(archive_filename, prompt)
