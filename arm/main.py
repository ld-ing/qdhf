# Adapted from https://github.com/icaros-usc/dqd/blob/main/experiments/arm/arm.py

import csv
import gc
import os
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from alive_progress import alive_bar
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter, ImprovementEmitter, IsoLineEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap
from utils import fit_ae, fit_dis_embed, fit_pca

matplotlib.use("Agg")
from matplotlib import pyplot as plt

sys.path.append(".")


def evaluate_grasp(
    joint_angles, method, metadata=None, device="cpu", return_features=False
):
    objs = -np.var(joint_angles, axis=1)
    # Remap the objective from [-1, 0] to [0, 100]
    objs = (objs + 1.0) * 100.0

    if metadata is None:
        metadata = {}

    cum_theta = np.cumsum(joint_angles, axis=1)
    x_pos = np.cos(cum_theta)
    y_pos = np.sin(cum_theta)
    features = np.concatenate((x_pos, y_pos), axis=1)

    if "dis_embed" in metadata:
        if metadata["dis_embed"] is not None:
            with torch.no_grad():
                features = (
                    metadata["dis_embed"](
                        torch.tensor(joint_angles, dtype=torch.float32).to(device)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

    if method is None:
        if return_features:
            return objs, features
        else:
            return objs
    elif method in ["qd", "gthf"]:
        link_lengths = np.ones(joint_angles.shape[1])
        # theta_1, theta_1 + theta_2, ...
        cum_theta = np.cumsum(joint_angles, axis=1)
        # l_1 * cos(theta_1), l_2 * cos(theta_1 + theta_2), ...
        x_pos = link_lengths[None] * np.cos(cum_theta)
        # l_1 * sin(theta_1), l_2 * sin(theta_1 + theta_2), ...
        y_pos = link_lengths[None] * np.sin(cum_theta)

        if method == "qd":
            measures = np.concatenate(
                (
                    np.sum(x_pos, axis=1, keepdims=True),
                    np.sum(y_pos, axis=1, keepdims=True),
                ),
                axis=1,
            )
        elif method == "gthf":
            measures = np.concatenate(
                (
                    np.cumsum(x_pos, axis=1),
                    np.cumsum(y_pos, axis=1),
                ),
                axis=1,
            )
    elif method == "pca":
        assert "pca" in metadata
        measures = metadata["pca"].transform(features)
    elif method == "ae":
        assert "ae" in metadata
        with torch.no_grad():
            measures = (
                metadata["ae"](torch.tensor(features, dtype=torch.float32).to(device))
                .detach()
                .cpu()
                .numpy()
            )
    elif method == "qdhf":
        measures = features
    else:
        raise NotImplementedError(f"Unknown method: {method}")

    if return_features:
        return objs, measures, features
    else:
        return objs, measures


def create_optimizer(
    method,
    sols,
    features=None,
    metadata=None,
    algorithm="map_elites",
    device="cpu",
    gt_bounds=None,
    seed=None,
    archive_bounds=None,
):
    """Creates archive and optimizer based on the algorithm name."""

    batch_size = 100
    num_emitters = 1

    if method == "qd":
        assert gt_bounds is not None
        objs, measures = evaluate_grasp(sols, method, metadata)
        if archive_bounds is None:
            archive_bounds = gt_bounds
    elif method == "pca":
        assert features is not None
        objs, measures = evaluate_grasp(sols, method, metadata, device)
        if archive_bounds is None:
            archive_bounds = np.array(
                [np.min(measures, axis=0), np.max(measures, axis=0)]
            ).T
    elif method == "ae":
        assert features is not None
        objs, measures = evaluate_grasp(sols, method, metadata, device)
        if archive_bounds is None:
            archive_bounds = np.array(
                [np.min(measures, axis=0), np.max(measures, axis=0)]
            ).T
    elif method == "qdhf":
        objs, measures = evaluate_grasp(sols, method, metadata, device)
        if archive_bounds is None:
            archive_bounds = np.array(
                [np.min(measures, axis=0), np.max(measures, axis=0)]
            ).T
    else:
        raise NotImplementedError(f"Unknown method: {method}")

    archive = GridArchive((50, 50), archive_bounds, seed)
    archive.initialize(solution_dim=len(sols[0]))
    # Add each solution to the archive.
    for i in range(len(sols)):
        archive.add(sols[i], objs[i], measures[i])
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

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

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

    link_lengths = np.ones(dim)
    max_bound = np.sum(link_lengths)
    gt_archive_bounds = np.array([[-max_bound, max_bound], [-max_bound, max_bound]])

    archive = None

    # Create a gt_archive to evaluate all the solutions.
    gt_archive_all = GridArchive((50, 50), gt_archive_bounds, seed=seed)
    gt_archive_all.initialize(dim)

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
                    # Sample initial population.
                    all_sols = np.random.uniform(
                        low=-np.pi, high=np.pi, size=(init_pop, dim)
                    )

                    # Initialize the dis embed.
                    if use_dis_embed:
                        inputs = np.random.uniform(
                            low=-np.pi, high=np.pi, size=(n_pref_data * 3, dim)
                        )
                        _, gt_measures = evaluate_grasp(inputs, method="qd")
                        dis_embed_data = inputs.reshape((n_pref_data, 3, dim))
                        dis_embed_gt_measures = gt_measures.reshape((n_pref_data, 3, 2))
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
                    # if len(outliers) > 0:
                    #     sols = np.concatenate((sols, np.array(outliers)), axis=0)

                    # Update the dis embed.
                    if use_dis_embed:
                        additional_inputs = [
                            all_sols[np.random.choice(all_sols.shape[0], 3)]
                            for _ in range(n_pref_data)
                        ]
                        additional_inputs = np.array(additional_inputs)
                        _, additional_gt_measures = evaluate_grasp(
                            additional_inputs.reshape(n_pref_data * 3, dim),
                            method="qd",
                        )
                        additional_gt_measures = additional_gt_measures.reshape(
                            n_pref_data, 3, 2
                        )
                        dis_embed_data = np.concatenate(
                            (dis_embed_data, additional_inputs), axis=0
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

                metadata = {"dis_embed": dis_embed}
                _, all_features = evaluate_grasp(
                    all_sols,
                    method=None,
                    metadata=metadata,
                    device=device,
                    return_features=True,
                )

                if method == "pca":
                    pca = fit_pca(all_features)
                    metadata["pca"] = pca
                elif method == "ae":
                    ae = fit_ae(all_features, device=device)
                    metadata["ae"] = ae

                archive, optimizer, metadata = create_optimizer(
                    method,
                    all_sols,
                    all_features,
                    metadata,
                    algorithm=algorithm,
                    device=device,
                    gt_bounds=gt_archive_bounds,
                    seed=seed,
                )
                archive_bounds = metadata["archive_bounds"]

                _objs, _gt_measures = evaluate_grasp(all_sols, method="qd")
                for i in range(len(all_sols)):
                    gt_archive_all.add(all_sols[i], _objs[i], _gt_measures[i])

            sols = optimizer.ask()
            objs, measures, features = evaluate_grasp(
                sols, method, metadata, device, return_features=True
            )
            best = max(best, max(objs))

            _objs, _gt_measures = evaluate_grasp(sols, method="qd")
            for i in range(len(sols)):
                gt_archive_all.add(sols[i], _objs[i], _gt_measures[i])

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
                _, all_features = evaluate_grasp(
                    all_sols,
                    method=None,
                    metadata=metadata,
                    device=device,
                    return_features=True,
                )
                archive, optimizer, metadata = create_optimizer(
                    method,
                    all_sols,
                    all_features,
                    metadata,
                    algorithm=algorithm,
                    device=device,
                    gt_bounds=gt_archive_bounds,
                    seed=seed,
                    archive_bounds=archive_bounds,
                )
            else:
                optimizer.tell(objs, measures)

            non_logging_time += time.time() - itr_start
            progress()

            # Always save on the final iteration.
            final_itr = itr == itrs

            # Update the summary statistics for the archive
            if (itr > 0 and itr % log_freq == 0) or final_itr:
                # Create a gt_archive to evaluate the current archive.
                gt_archive = GridArchive((50, 50), gt_archive_bounds, seed=seed)
                gt_archive.initialize(dim)
                sols = archive.data()[0]
                objs, gt_measures = evaluate_grasp(sols, method="qd")
                for i in range(len(sols)):
                    gt_archive.add(sols[i], objs[i], gt_measures[i])

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


def arm_main(
    method,
    trial_id=0,
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
    arm_main(method="qd", trial_id=trial_id)

    # AURORA
    for online_finetune in [False, True]:
        arm_main(
            method="pca",
            trial_id=trial_id,
            online_finetune=online_finetune,
        )
        arm_main(
            method="ae",
            trial_id=trial_id,
            online_finetune=online_finetune,
        )

    # QDHF
    n_pref_data = 1000
    arm_main(
        method="qdhf",
        trial_id=trial_id,
        use_dis_embed=True,
        n_pref_data=n_pref_data,
        online_finetune=False,
    )
    arm_main(
        method="qdhf",
        trial_id=trial_id,
        use_dis_embed=True,
        n_pref_data=n_pref_data // 4,
        online_finetune=True,
    )
