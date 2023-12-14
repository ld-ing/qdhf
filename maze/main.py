# Adapted from https://github.com/adaptive-intelligent-robotics/Kheperax/blob/main/examples/me_training.py

import csv
import functools
import gc
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import torch
from alive_progress import alive_bar
from kheperax.task import KheperaxConfig, KheperaxTask
from qdax.core.emitters.mutation_operators import isoline_variation
from ribs.archives import GridArchive
from ribs.visualize import grid_archive_heatmap
from utils import fit_ae, fit_dis_embed, fit_pca

matplotlib.use("Agg")
from matplotlib import pyplot as plt

sys.path.append(".")

grid_shape = (50, 50)
episode_length = 250
mlp_policy_hidden_layer_sizes = (8,)
batch_size = 200


def evaluate_maze(
    inputs,
    method,
    scoring_fn,
    metadata=None,
    device="cpu",
    return_features=False,
    random_key=None,
):
    if metadata is None:
        metadata = {}

    fitnesses, descriptors, extra_scores, random_key = scoring_fn(inputs, random_key)
    fitnesses = (fitnesses + 0.3) / 0.3 * 100

    features = extra_scores["transitions"].state_desc  # batch, 250, 2
    features = features.reshape(features.shape[0], 500)  # batch, 500

    if "dis_embed" in metadata:
        if metadata["dis_embed"] is not None:
            with torch.no_grad():
                features = (
                    metadata["dis_embed"](
                        torch.tensor(np.array(features), dtype=torch.float32).to(device)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

    if method is None:
        if return_features:
            return fitnesses, features
        else:
            return fitnesses
    elif method == "qd":
        measures = descriptors
    elif method == "pca":
        assert "pca" in metadata
        measures = metadata["pca"].transform(features)
    elif method == "ae":
        assert "ae" in metadata
        with torch.no_grad():
            measures = (
                metadata["ae"](
                    torch.tensor(np.array(features), dtype=torch.float32).to(device)
                )
                .detach()
                .cpu()
                .numpy()
            )
    elif method == "qdhf":
        measures = features
    else:
        raise NotImplementedError(f"Unknown method: {method}")

    if return_features:
        return fitnesses, measures, features
    else:
        return fitnesses, measures


def create_optimizer(
    method,
    sols,
    features=None,
    scoring_fn=None,
    metadata=None,
    algorithm="map_elites",
    device="cpu",
    gt_bounds=None,
    seed=None,
    archive_bounds=None,
    random_key=None,
):
    """Creates archive and optimizer based on the algorithm name."""

    num_emitters = 1

    if method == "qd":
        assert gt_bounds is not None
        objs, measures = evaluate_maze(
            sols, method, scoring_fn, metadata, random_key=random_key
        )
        if archive_bounds is None:
            archive_bounds = gt_bounds
    elif method == "pca":
        assert features is not None
        objs, measures = evaluate_maze(
            sols, method, scoring_fn, metadata, device, random_key=random_key
        )
        if archive_bounds is None:
            archive_bounds = np.array(
                [np.min(measures, axis=0), np.max(measures, axis=0)]
            ).T
    elif method == "ae":
        assert features is not None
        objs, measures = evaluate_maze(
            sols, method, scoring_fn, metadata, device, random_key=random_key
        )
        if archive_bounds is None:
            archive_bounds = np.array(
                [np.min(measures, axis=0), np.max(measures, axis=0)]
            ).T
    elif method == "qdhf":
        objs, measures = evaluate_maze(
            sols, method, scoring_fn, metadata, device, random_key=random_key
        )
        if archive_bounds is None:
            archive_bounds = np.array(
                [np.min(measures, axis=0), np.max(measures, axis=0)]
            ).T
    else:
        raise NotImplementedError(f"Unknown method: {method}")

    sols = batch_to_list(sols)
    archive = GridArchive(grid_shape, archive_bounds, seed, dtype=object)
    archive.initialize(solution_dim=1)
    # Add each solution to the archive.
    for i in range(len(sols)):
        archive.add(sols[i], objs[i], measures[i])
    metadata["archive_bounds"] = archive_bounds

    return archive, metadata


def emit(archive, variation_fn, random_key):
    pool = archive.data()[0]
    pool = list_to_batch(pool)
    random_key, subkey = jax.random.split(random_key)
    x1 = jax.tree_util.tree_map(
        lambda x: jax.random.choice(subkey, x, shape=(batch_size,)),
        pool,
    )
    random_key, subkey = jax.random.split(random_key)
    x2 = jax.tree_util.tree_map(
        lambda x: jax.random.choice(subkey, x, shape=(batch_size,)),
        pool,
    )
    population, random_key = variation_fn(x1, x2, random_key)
    return population, random_key


def batch_to_list(batch):
    return [
        jax.tree_util.tree_map(
            lambda x: x[i],
            batch,
        )
        for i in range(batch_size)
    ]


def list_to_batch(l):
    l = [i[0] for i in l]
    return jax.tree_util.tree_map(
        lambda *v: jnp.stack(v),
        *l,
    )


def save_heatmap(archive, heatmap_path):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive or CVTArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    fig = plt.figure(figsize=(8, 6), dpi=300)
    grid_archive_heatmap(archive, vmin=0, vmax=100)
    plt.tight_layout()
    fig.savefig(heatmap_path)

    plt.clf()
    plt.close(fig)
    gc.collect()


def run_experiment(
    method,
    trial_id,
    dim=1000,
    init_pop=1000,
    itrs=10000,
    outdir="logs",
    log_freq=1,
    log_arch_freq=1000,
    seed=None,
    use_dis_embed=False,
    n_pref_data=1000,
    online_finetune=False,
    incre_bounds=False,
):
    algorithm = "map_elites"
    device = "cpu"

    batch_size = 200
    num_evaluations = int(2e5)
    num_iterations = num_evaluations // batch_size

    if seed is None:
        seed = 42

    # Init a random key
    np.random.seed(seed)
    torch.manual_seed(seed)
    random_key = jax.random.PRNGKey(seed)
    random_key, subkey = jax.random.split(random_key)

    # Define Task configuration
    config_kheperax = KheperaxConfig.get_default()

    # Example of modification of the robots attributes (same thing could be done with the maze)
    config_kheperax.robot = config_kheperax.robot.replace(
        lasers_return_minus_one_if_out_of_range=True
    )

    # Create Kheperax Task.
    (
        env,
        policy_network,
        scoring_fn,
    ) = KheperaxTask.create_default_task(
        config_kheperax,
        random_key=subkey,
    )

    # Define emitter
    iso_sigma = 0.2
    line_sigma = 0.0
    variation_fn = functools.partial(
        isoline_variation,
        iso_sigma=iso_sigma,
        line_sigma=line_sigma,
    )

    # Create a directory for this specific trial.
    s_logdir = os.path.join(outdir, f"{algorithm}_trial_{trial_id}")
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
            "QD-Score (search)",
            "Coverage (search)",
            "Maximum (search)",
            "Average (search)",
            "QD-Score (fit)",
            "Coverage (fit)",
            "Maximum (fit)",
            "Average (fit)",
            "DisEmbed Acc",
        ]
        writer.writerow(col_names)

    min_bd, max_bd = env.behavior_descriptor_limits
    gt_archive_bounds = np.array([min_bd, max_bd]).T

    archive = None

    # Create a gt_archive to evaluate all the solutions.
    gt_archive_all = GridArchive(grid_shape, gt_archive_bounds, seed=seed)
    gt_archive_all.initialize(1)

    best = 0.0
    non_logging_time = 0.0
    with alive_bar(itrs) as progress:
        for itr in range(1, itrs + 1):
            itr_start = time.time()

            # Update archive.
            if online_finetune:
                update_schedule = [1, 101, 251, 501]
            else:
                update_schedule = [1]
            if itr in update_schedule:
                if archive is None:
                    # initialising first variables for Map-Elites init
                    random_key, subkey = jax.random.split(random_key)
                    keys = jax.random.split(subkey, num=batch_size)
                    fake_batch = jnp.zeros(shape=(batch_size, env.observation_size))
                    all_sols = jax.vmap(policy_network.init)(keys, fake_batch)

                    # Initialize the dis embed.
                    if use_dis_embed:
                        random_key, subkey = jax.random.split(random_key)
                        keys = jax.random.split(subkey, num=n_pref_data * 3)
                        fake_batch = jnp.zeros(
                            shape=(n_pref_data * 3, env.observation_size)
                        )
                        inputs = jax.vmap(policy_network.init)(keys, fake_batch)

                        _, gt_measures, features = evaluate_maze(
                            inputs,
                            method="qd",
                            scoring_fn=scoring_fn,
                            return_features=True,
                            random_key=random_key,
                        )

                        dis_embed_data = features.reshape(n_pref_data, 3, -1)
                        dis_embed_gt_measures = gt_measures.reshape(n_pref_data, 3, -1)
                        dis_embed, dis_embed_acc = fit_dis_embed(
                            dis_embed_data,
                            dis_embed_gt_measures,
                            latent_dim=2,
                            seed=seed,
                        )
                    else:
                        dis_embed = None
                        dis_embed_acc = -1
                else:
                    all_sols = archive.data()[0]

                    # Update the dis embed.
                    if use_dis_embed:
                        additional_inputs = [
                            all_sols[np.random.randint(len(all_sols))]
                            for _ in range(n_pref_data * 3)
                        ]
                        additional_inputs = list_to_batch(additional_inputs)
                        (
                            _,
                            additional_gt_measures,
                            additional_features,
                        ) = evaluate_maze(
                            additional_inputs,
                            method="qd",
                            scoring_fn=scoring_fn,
                            return_features=True,
                            random_key=random_key,
                        )
                        additional_features = additional_features.reshape(
                            n_pref_data, 3, -1
                        )
                        additional_gt_measures = additional_gt_measures.reshape(
                            n_pref_data, 3, -1
                        )
                        dis_embed_data = np.concatenate(
                            (dis_embed_data, additional_features), axis=0
                        )
                        dis_embed_gt_measures = np.concatenate(
                            (dis_embed_gt_measures, additional_gt_measures), axis=0
                        )
                        dis_embed, dis_embed_acc = fit_dis_embed(
                            dis_embed_data,
                            dis_embed_gt_measures,
                            latent_dim=2,
                            seed=seed,
                        )

                    all_sols = list_to_batch(all_sols)

                metadata = {"dis_embed": dis_embed}
                _, all_features = evaluate_maze(
                    all_sols,
                    method=None,
                    scoring_fn=scoring_fn,
                    metadata=metadata,
                    device=device,
                    return_features=True,
                    random_key=random_key,
                )

                if method == "pca":
                    pca = fit_pca(all_features)
                    metadata["pca"] = pca
                elif method == "ae":
                    ae = fit_ae(all_features, device=device)
                    metadata["ae"] = ae

                archive, metadata = create_optimizer(
                    method,
                    all_sols,
                    all_features,
                    scoring_fn,
                    metadata,
                    algorithm=algorithm,
                    device=device,
                    gt_bounds=gt_archive_bounds,
                    seed=seed,
                    random_key=random_key,
                )
                archive_bounds = metadata["archive_bounds"]

                _objs, _gt_measures = evaluate_maze(
                    all_sols,
                    method="qd",
                    scoring_fn=scoring_fn,
                    random_key=random_key,
                )
                for i in range(len(_objs)):
                    gt_archive_all.add(1, _objs[i], _gt_measures[i])

            sols, random_key = emit(archive, variation_fn, random_key)
            objs, measures, features = evaluate_maze(
                sols,
                method,
                scoring_fn,
                metadata,
                device,
                return_features=True,
                random_key=random_key,
            )
            best = max(best, max(objs))

            _objs, _gt_measures = evaluate_maze(
                sols,
                method="qd",
                scoring_fn=scoring_fn,
                random_key=random_key,
            )
            for i in range(len(_objs)):
                gt_archive_all.add(1, _objs[i], _gt_measures[i])

            # Check if any solutions are out of bound.
            # If so, update the archive with new bounds.
            update_archive = False
            if incre_bounds:
                if np.min(measures[:, 0]) < archive_bounds[0, 0]:
                    archive_bounds[0, 0] = np.min(measures[:, 0])
                    update_archive = True
                if np.max(measures[:, 0]) > archive_bounds[0, 1]:
                    archive_bounds[0, 1] = np.max(measures[:, 0])
                    update_archive = True
                if np.min(measures[:, 1]) < archive_bounds[1, 0]:
                    archive_bounds[1, 0] = np.min(measures[:, 1])
                    update_archive = True
                if np.max(measures[:, 1]) > archive_bounds[1, 1]:
                    archive_bounds[1, 1] = np.max(measures[:, 1])
                    update_archive = True

            if update_archive:
                all_sols = archive.data()[0]
                all_sols = np.concatenate((all_sols, sols), axis=0)
                _, all_features = evaluate_maze(
                    all_sols,
                    method=None,
                    scoring_fn=scoring_fn,
                    metadata=metadata,
                    device=device,
                    return_features=True,
                    random_key=random_key,
                )
                archive, metadata = create_optimizer(
                    method,
                    all_sols,
                    all_features,
                    scoring_fn,
                    metadata,
                    algorithm=algorithm,
                    device=device,
                    gt_bounds=gt_archive_bounds,
                    seed=seed,
                    archive_bounds=archive_bounds,
                )
            else:
                # Add each solution to the archive.
                sols = batch_to_list(sols)
                for i in range(len(sols)):
                    archive.add(sols[i], objs[i], measures[i])

            non_logging_time += time.time() - itr_start
            progress()

            # Always save on the final iteration.
            final_itr = itr == itrs

            # Update the summary statistics for the archive
            if (itr > 0 and itr % log_freq == 0) or final_itr:
                # Create a gt_archive to evaluate the solutions.
                gt_archive = GridArchive(grid_shape, gt_archive_bounds, seed=seed)
                gt_archive.initialize(solution_dim=1)
                sols = archive.data()[0]
                objs, gt_measures = evaluate_maze(
                    list_to_batch(sols),
                    method="qd",
                    scoring_fn=scoring_fn,
                    random_key=random_key,
                )
                for i in range(len(sols)):
                    gt_archive.add(1, objs[i], gt_measures[i])

                sum_obj = 0
                num_filled = 0
                num_bins = archive.bins
                for sol, obj, beh, idx, meta in zip(*gt_archive.data()):
                    num_filled += 1
                    sum_obj += obj
                qd_score = sum_obj / num_bins
                average = sum_obj / num_filled
                coverage = 100.0 * num_filled / num_bins
                data = [itr, qd_score, coverage, best, average]

                sum_obj = 0
                num_filled = 0
                num_bins = archive.bins
                for sol, obj, beh, idx, meta in zip(*gt_archive_all.data()):
                    num_filled += 1
                    sum_obj += obj
                qd_score = sum_obj / num_bins
                average = sum_obj / num_filled
                coverage = 100.0 * num_filled / num_bins
                data += [qd_score, coverage, best, average]

                sum_obj = 0
                num_filled = 0
                num_bins = archive.bins
                for sol, obj, beh, idx, meta in zip(*archive.data()):
                    num_filled += 1
                    sum_obj += obj
                qd_score = sum_obj / num_bins
                average = sum_obj / num_filled
                coverage = 100.0 * num_filled / num_bins
                data += [qd_score, coverage, best, average]
                data += [dis_embed_acc]

                with open(summary_filename, "a") as summary_file:
                    writer = csv.writer(summary_file)
                    writer.writerow(data)

            if itr % log_arch_freq == 0 or final_itr:
                # Save a full archive for analysis.
                # df = archive.as_pandas(include_solutions=final_itr)
                # df.to_pickle(os.path.join(s_logdir, f"{method}_archive_{itr:08d}.pkl"))

                # Save a heatmap image to observe how the trial is doing.
                file_name = log_file_name + f"_heatmap_{itr:08d}.png"
                save_heatmap(
                    archive,
                    os.path.join(s_logdir, file_name),
                )

                file_name = log_file_name + f"_gtheatmap_{itr:08d}.png"
                save_heatmap(
                    gt_archive,
                    os.path.join(s_logdir, file_name),
                )

                file_name = log_file_name + f"_gtheatmapall_{itr:08d}.png"
                save_heatmap(
                    gt_archive_all,
                    os.path.join(s_logdir, file_name),
                )

    print(log_file_name, "| QD score:", data[1], "Coverage:", data[2])
    print()

    del env, policy_network, scoring_fn, archive, gt_archive, gt_archive_all


def arm_main(
    trial_id,
    method,
    dim=10,
    init_pop=100,
    itrs=1000,
    outdir="logs",
    log_freq=20,
    log_arch_freq=100,
    use_dis_embed=False,
    n_pref_data=1000,
    online_finetune=False,
    incre_bounds=False,
):
    """Experimental tool for the planar robotic arm experiments."""

    # Create a shared logging directory for the experiments for this algorithm.
    outdir = Path(outdir)
    if not outdir.is_dir():
        outdir.mkdir()

    # Run an experiment as a separate process to run all exps in parallel.
    run_experiment(
        method,
        trial_id,
        dim=dim,
        init_pop=init_pop,
        itrs=itrs,
        outdir=outdir,
        log_freq=log_freq,
        log_arch_freq=log_arch_freq,
        seed=trial_id,
        use_dis_embed=use_dis_embed,
        n_pref_data=n_pref_data,
        online_finetune=online_finetune,
        incre_bounds=incre_bounds,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        trial_id = int(sys.argv[1])
    else:
        trial_id = 0

    # QD-GT
    arm_main(trial_id, method="qd")

    # AURORA
    for method in ["pca", "ae"]:
        for online_finetune in [True, False]:
            arm_main(trial_id, method=method, online_finetune=online_finetune)

    # QDHF
    n_pref_data = 200
    for online_finetune in [False, True]:
        data = n_pref_data if not online_finetune else n_pref_data // 4
        arm_main(
            trial_id,
            method="qdhf",
            use_dis_embed=True,
            n_pref_data=data,
            online_finetune=online_finetune,
        )
