# Adapted from https://github.com/icaros-usc/dqd/blob/main/experiments/lsi_clip/lsi.py

import csv
import gc
import os
import sys
import time
from pathlib import Path

import clip
import matplotlib
import numpy as np
import torch
from alive_progress import alive_bar
from collage import make_archive_collage
from diffusers import StableDiffusionPipeline
from dreamsim import dreamsim
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter, ImprovementEmitter, IsoLineEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap
from lsi_utils import calc_pairwise_dis, fit_ae, fit_dis_embed, fit_pca

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")

grid_shape = (20, 20)
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
batch_size = 8
init_pop = 400


def compute_clip_scores(
    device, clip_model, clip_preprocess, imgs, text, return_features=False
):
    img_tensor = torch.stack([clip_preprocess(img) for img in imgs]).to(device)
    tokenized_text = clip.tokenize([text]).to(device)
    img_logits, _text_logits = clip_model(img_tensor, tokenized_text)
    img_logits = img_logits.detach().cpu().numpy().astype(np.float32)[:, 0]
    img_logits = 1 / img_logits * 100
    # Remap the objective from minimizing [0, 10] to maximizing [0, 100]
    img_logits = (10.0 - img_logits) * 10.0

    if return_features:
        img_features = clip_model.encode_image(img_tensor)
        return img_logits, img_features
    else:
        return img_logits


def evaluate_lsi(
    latents,
    method,
    metadata,
    device="cpu",
    return_features=False,
    clip_features=None,
    dreamsim_features=None,
    objs=None,
):
    # print(latents.shape)  # torch.Size([10, 4, 64, 64])
    assert "pipe" in metadata
    assert "prompt" in metadata
    assert "dreamsim_model" in metadata
    assert "dreamsim_preprocess" in metadata
    assert "clip_model" in metadata
    assert "clip_preprocess" in metadata

    if clip_features is None or objs is None:
        images = metadata["pipe"](
            metadata["prompt"],
            num_images_per_prompt=latents.shape[0],
            latents=latents,
            # num_inference_steps=1,  # for test
        ).images

        objs, clip_features = compute_clip_scores(
            device,
            metadata["clip_model"],
            metadata["clip_preprocess"],
            images,
            metadata["prompt"],
            return_features=True,
        )

        images = torch.cat([metadata["dreamsim_preprocess"](img) for img in images]).to(
            device
        )
        dreamsim_features = metadata["dreamsim_model"].embed(images)

    if method is None:
        if return_features:
            return objs, clip_features, dreamsim_features
        else:
            return objs

    if method == "pca":
        assert "pca" in metadata
        measures = metadata["pca"].transform(clip_features.detach().cpu().numpy())
    elif method == "ae":
        assert "ae" in metadata
        with torch.no_grad():
            measures = metadata["ae"](clip_features).detach().cpu().numpy()
    elif method == "qdhf":
        assert "dis_embed" in metadata
        assert metadata["dis_embed"] is not None
        with torch.no_grad():
            measures = (
                metadata["dis_embed"](clip_features.to(torch.float32))
                .detach()
                .cpu()
                .numpy()
            )
    else:
        raise NotImplementedError(f"Unknown method: {method}")

    if return_features:
        return objs, measures, clip_features, dreamsim_features
    else:
        return objs, measures


def tensor_to_list(tensor):
    sols = tensor.detach().cpu().numpy().astype(np.float32)
    return sols.reshape(sols.shape[0], -1)


def list_to_tensor(list_, device):
    sols = np.array(list_).reshape(len(list_), 4, 64, 64)  # hard-coded for now
    return torch.tensor(sols, dtype=torch_dtype, device=device)


def create_optimizer(
    method,
    sols,
    clip_features,
    dreamsim_features,
    objs,
    metadata=None,
    algorithm="map_elites",
    device="cpu",
    seed=None,
    archive_bounds=None,
):
    """Creates archive and optimizer based on the algorithm name."""

    num_emitters = 1

    if method == "pca":
        objs, measures = evaluate_lsi(
            sols, method, metadata, device, clip_features=clip_features, objs=objs
        )
        if archive_bounds is None:
            archive_bounds = np.array(
                [np.min(measures, axis=0), np.max(measures, axis=0)]
            ).T
    elif method == "ae":
        objs, measures = evaluate_lsi(
            sols,
            method,
            metadata,
            device,
            clip_features=clip_features,
            objs=objs,
        )
        if archive_bounds is None:
            archive_bounds = np.array(
                [np.min(measures, axis=0), np.max(measures, axis=0)]
            ).T
    elif method == "qdhf":
        objs, measures = evaluate_lsi(
            sols,
            method,
            metadata,
            device,
            clip_features=clip_features.float(),
            objs=objs,
        )
        if archive_bounds is None:
            archive_bounds = np.array(
                [np.min(measures, axis=0), np.max(measures, axis=0)]
            ).T
    else:
        raise NotImplementedError(f"Unknown method: {method}")

    sols = tensor_to_list(sols)
    archive = GridArchive(grid_shape, archive_bounds, seed)
    archive.initialize(solution_dim=len(sols[0]))

    # Add each solution to the archive.
    dreamsim_features = dreamsim_features.detach().cpu().numpy()
    for i in range(len(sols)):
        archive.add(sols[i], objs[i], measures[i], dreamsim_features[i])
    metadata["archive_bounds"] = archive_bounds

    # Create emitters. Each emitter needs a different seed, so that they do not
    # all do the same thing.
    initial_sol = archive.get_random_elite()[0]
    emitter_seeds = (
        [None] * num_emitters
        if seed is None
        else list(range(seed, seed + num_emitters))
    )
    if algorithm in ["map_elites"]:
        emitters = [
            GaussianEmitter(archive, initial_sol, 0.1, batch_size=batch_size, seed=s)
            for s in emitter_seeds
        ]
    elif algorithm in ["map_elites_line"]:
        emitters = [
            IsoLineEmitter(
                archive,
                initial_sol,
                iso_sigma=0.1,
                line_sigma=0.2,
                batch_size=batch_size,
                seed=s,
            )
            for s in emitter_seeds
        ]
    elif algorithm in ["cma_me_imp"]:
        emitters = [
            ImprovementEmitter(archive, initial_sol, 0.1, batch_size=batch_size, seed=s)
            for s in emitter_seeds
        ]

    return archive, Optimizer(archive, emitters, init_archive=False), metadata


def save_heatmap(archive, heatmap_path):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive or CVTArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    fig = plt.figure(figsize=(8, 6), dpi=300)
    grid_archive_heatmap(archive, vmin=0.0, vmax=100.0)
    plt.tight_layout()
    fig.savefig(heatmap_path)

    plt.clf()
    plt.close(fig)
    gc.collect()


def run_experiment(
    method,
    prompt,
    itrs=200,
    outdir="logs",
    log_freq=1,
    log_arch_freq=20,
    seed=123,
    use_dis_embed=False,
    n_pref_data=100,
    online_finetune=False,
    incre_bounds=False,
):
    algorithm = "map_elites"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create a directory for this specific trial.
    s_logdir = os.path.join(outdir, f"{algorithm}_{prompt}")
    logdir = Path(s_logdir)
    if not logdir.is_dir():
        logdir.mkdir()

    # Create a new summary file
    if use_dis_embed:
        log_file_name = f"{method}(n={n_pref_data * 4 if online_finetune else n_pref_data})|{'online' if online_finetune else 'offline'}|{'incre' if incre_bounds else 'fixed'}"
    else:
        log_file_name = f"{method}|{'online' if online_finetune else 'offline'}|{'incre' if incre_bounds else 'fixed'}"

    summary_filename = os.path.join(
        s_logdir,
        log_file_name + "_summary.csv",
    )

    if os.path.exists(summary_filename):
        os.remove(summary_filename)
    with open(summary_filename, "w") as summary_file:
        writer = csv.writer(summary_file)
        col_names = [
            "Iteration",
            "QD-Score",
            "Coverage",
            "Maximum",
            "Average",
            "DisEmbed Acc",
            "Pair Dis Mean",
            "Pair Dis Std",
        ]
        writer.writerow(col_names)

    archive = None

    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,  # for faster inference
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    num_images_per_prompt = batch_size  # pop size
    num_channels_latents = pipe.unet.config.in_channels
    height = pipe.unet.config.sample_size
    width = pipe.unet.config.sample_size

    latents_shape = (
        num_images_per_prompt,
        num_channels_latents,
        height,
        width,
    )  # torch.Size([1, 4, 64, 64])

    dreamsim_model, dreamsim_preprocess = dreamsim(
        pretrained=True, dreamsim_type="dino_vitb16", device=device
    )

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    metadata = {
        "pipe": pipe,
        "prompt": prompt,
        "dreamsim_model": dreamsim_model,
        "dreamsim_preprocess": dreamsim_preprocess,
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,
    }

    best = 0.0
    non_logging_time = 0.0
    with alive_bar(itrs) as progress:
        for itr in range(1, itrs + 1):
            itr_start = time.time()

            # Update archive.
            if online_finetune:
                update_schedule = [1, 21, 51, 101]
            else:
                update_schedule = [1]
            if itr in update_schedule:
                if archive is None:
                    all_sols = []
                    all_clip_features = []
                    all_dreamsim_features = []
                    all_objs = []

                    n_batches = init_pop // batch_size
                    for _ in range(n_batches):
                        sols = torch.randn(
                            latents_shape, device=device, dtype=torch_dtype
                        )
                        objs, clip_features, dreamsim_features = evaluate_lsi(
                            sols, None, metadata, device=device, return_features=True
                        )
                        all_sols.append(sols)
                        all_clip_features.append(clip_features)
                        all_dreamsim_features.append(dreamsim_features)
                        all_objs.append(objs)
                    all_sols = torch.concat(all_sols, dim=0)
                    all_clip_features = torch.concat(
                        all_clip_features, dim=0
                    )  # init_pop, dim
                    all_dreamsim_features = torch.concat(all_dreamsim_features, dim=0)
                    all_objs = np.concatenate(all_objs, axis=0)

                    # Initialize the dis embed.
                    if use_dis_embed:
                        additional_features = []
                        additional_labels = []
                        for _ in range(n_pref_data):
                            idx = np.random.choice(all_sols.shape[0], 3)
                            additional_features.append(all_clip_features[idx])
                            additional_labels.append(all_dreamsim_features[idx])
                        additional_features = torch.concat(additional_features, dim=0)
                        additional_labels = torch.concat(additional_labels, dim=0)
                        dis_embed_data = additional_features.reshape(n_pref_data, 3, -1)
                        dis_embed_label = additional_labels.reshape(n_pref_data, 3, -1)
                        dis_embed, dis_embed_acc = fit_dis_embed(
                            dis_embed_data,
                            dis_embed_label,
                            latent_dim=2,
                            seed=seed,
                            device=device,
                        )
                    else:
                        dis_embed = None
                        dis_embed_acc = -1
                else:
                    all_sols = list_to_tensor(archive.data()[0], device)
                    n_batches = np.ceil(len(all_sols) / batch_size).astype(int)
                    all_clip_features = []
                    all_dreamsim_features = []
                    all_objs = []
                    for i in range(n_batches):
                        sols = all_sols[i * batch_size : (i + 1) * batch_size]
                        objs, clip_features, dreamsim_features = evaluate_lsi(
                            sols, None, metadata, device=device, return_features=True
                        )
                        all_clip_features.append(clip_features)
                        all_dreamsim_features.append(dreamsim_features)
                        all_objs.append(objs)
                    all_clip_features = torch.concat(
                        all_clip_features, dim=0
                    )  # n_pref_data * 3, dim
                    all_dreamsim_features = torch.concat(all_dreamsim_features, dim=0)
                    all_objs = np.concatenate(all_objs, axis=0)

                    # Update the dis embed.
                    if use_dis_embed:
                        additional_features = []
                        additional_labels = []
                        for _ in range(n_pref_data):
                            idx = np.random.choice(all_sols.shape[0], 3)
                            additional_features.append(all_clip_features[idx])
                            additional_labels.append(all_dreamsim_features[idx])
                        additional_features = torch.concat(additional_features, dim=0)
                        additional_labels = torch.concat(additional_labels, dim=0)
                        additional_embed_data = additional_features.reshape(
                            n_pref_data, 3, -1
                        )
                        additional_embed_label = additional_labels.reshape(
                            n_pref_data, 3, -1
                        )
                        dis_embed_data = torch.concat(
                            (dis_embed_data, additional_embed_data), axis=0
                        )
                        dis_embed_label = torch.concat(
                            (dis_embed_label, additional_embed_label), axis=0
                        )
                        dis_embed, dis_embed_acc = fit_dis_embed(
                            dis_embed_data,
                            dis_embed_label,
                            latent_dim=2,
                            seed=seed,
                            device=device,
                        )

                metadata["dis_embed"] = dis_embed

                if method == "pca":
                    pca = fit_pca(all_clip_features.detach().cpu().numpy())
                    metadata["pca"] = pca
                elif method == "ae":
                    ae = fit_ae(all_clip_features, device=device)
                    metadata["ae"] = ae

                archive, optimizer, metadata = create_optimizer(
                    method,
                    all_sols,
                    all_clip_features,
                    all_dreamsim_features,
                    all_objs,
                    metadata,
                    algorithm=algorithm,
                    device=device,
                    seed=seed,
                )
                archive_bounds = metadata["archive_bounds"]

            sols = optimizer.ask()
            sols = list_to_tensor(sols, device)
            objs, measures, clip_features, dreamsim_features = evaluate_lsi(
                sols, method, metadata, device, return_features=True
            )
            archive_meta = dreamsim_features.detach().cpu().numpy()
            best = max(best, max(objs))

            optimizer.tell(objs, measures, metadata=archive_meta)

            non_logging_time += time.time() - itr_start
            progress()

            # Always save on the final iteration.
            final_itr = itr == itrs

            # Update the summary statistics for the archive
            if (itr > 0 and itr % log_freq == 0) or final_itr:
                sum_obj = 0
                num_filled = 0
                num_bins = archive.bins
                for sol, obj, beh, idx, meta in zip(*archive.data()):
                    num_filled += 1
                    sum_obj += obj
                qd_score = sum_obj / num_bins
                average = sum_obj / num_filled
                coverage = 100.0 * num_filled / num_bins
                data = [itr, qd_score, coverage, best, average]
                data += [dis_embed_acc]

                archive_meta = np.stack(archive.data()[4], axis=0).astype(float)
                pairwise_dis_mean, pairwise_dis_std = calc_pairwise_dis(archive_meta)
                data += [pairwise_dis_mean, pairwise_dis_std]

                with open(summary_filename, "a") as summary_file:
                    writer = csv.writer(summary_file)
                    writer.writerow(data)

            if itr % log_arch_freq == 0 or final_itr:
                # Save a heatmap image to observe how the trial is doing.
                file_name = log_file_name + f"_heatmap_{itr:08d}.png"
                save_heatmap(
                    archive,
                    os.path.join(s_logdir, file_name),
                )

                # Save the archive.
                df = archive.as_pandas(
                    include_solutions=final_itr, include_metadata=final_itr
                )
                df.to_pickle(
                    os.path.join(s_logdir, log_file_name + f"_archive_{itr:08d}.pkl")
                )

    print(log_file_name, "| QD score:", data[1], "Coverage:", data[2])
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "a photo of an astronaut riding a horse on mars"

    # Create a shared logging directory for the experiments for this algorithm.
    outdir = Path("logs")
    if not outdir.is_dir():
        outdir.mkdir()

    run_experiment(
        "qdhf",
        prompt,
        use_dis_embed=True,
        n_pref_data=10000 // 4,
        online_finetune=True,
    )

    archive_filename = (
        f"logs/map_elites_{prompt}/qdhf(n=10000)|online|fixed_archive_00000200.pkl"
    )
    make_archive_collage(archive_filename, prompt)
