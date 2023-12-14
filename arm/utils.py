from sklearn.decomposition import PCA
import torch
from torch import nn
import numpy as np
import time


class DisEmbed(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=input_dim * 2, out_features=latent_dim),
        )

    def forward(self, x):
        x = torch.cumsum(x, 1)
        x1 = torch.cos(x)
        x2 = torch.sin(x)
        x = torch.cat([x1, x2], -1)
        x = self.enc(x)
        # x = torch.nn.functional.normalize(x, p=2, dim=1)  # l2 normalize
        return x

    def calc_dis(self, x1, x2):
        x1 = self.forward(x1)
        x2 = self.forward(x2)
        # return 1 - torch.sum(x1 * x2, -1)
        return torch.sum(torch.square(x1 - x2), -1)

    def triplet_delta_dis(self, ref, x1, x2):
        x1 = self.forward(x1)
        x2 = self.forward(x2)
        ref = self.forward(ref)
        # return torch.sum(ref * x2, -1) - torch.sum(ref * x1, -1)
        return torch.sum(torch.square(ref - x1), -1) - torch.sum(
            torch.square(ref - x2), -1
        )


def fit_dis_embed(
    inputs, gt_measures, latent_dim, batch_size=32, seed=None, device="cpu"
):
    t = time.time()
    model = DisEmbed(input_dim=inputs.shape[-1], latent_dim=latent_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = lambda y, delta_dis: torch.max(
        torch.tensor([0.0]), 0.05 - y * delta_dis
    ).mean()
    n_pref_data = inputs.shape[0]
    ref = inputs[:, 0]
    x1 = inputs[:, 1]
    x2 = inputs[:, 2]

    n_train = int(n_pref_data * 0.75)
    n_val = n_pref_data - n_train

    ref_train = ref[:n_train]
    x1_train = x1[:n_train]
    x2_train = x2[:n_train]
    ref_val = ref[n_train:]
    x1_val = x1[n_train:]
    x2_val = x2[n_train:]
    # ref_val = ref_train
    # x1_val = x1_train
    # x2_val = x2_train

    n_iters_per_epoch = max((n_train) // batch_size, 1)

    ref_gt_measures = gt_measures[:, 0]
    x1_gt_measures = gt_measures[:, 1]
    x2_gt_measures = gt_measures[:, 2]
    ref_gt_train = ref_gt_measures[:n_train]
    x1_gt_train = x1_gt_measures[:n_train]
    x2_gt_train = x2_gt_measures[:n_train]
    ref_gt_val = ref_gt_measures[n_train:]
    x1_gt_val = x1_gt_measures[n_train:]
    x2_gt_val = x2_gt_measures[n_train:]
    # ref_gt_val = ref_gt_train
    # x1_gt_val = x1_gt_train
    # x2_gt_val = x2_gt_train

    # best_acc = 0
    # counter = 0
    val_acc = []
    for epoch in range(1000):
        for _ in range(n_iters_per_epoch):
            idx = np.random.choice(n_train, batch_size)
            batch_ref = torch.tensor(ref_train[idx], dtype=torch.float32).to(device)
            batch1 = torch.tensor(x1_train[idx], dtype=torch.float32).to(device)
            batch2 = torch.tensor(x2_train[idx], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            delta_dis = model.triplet_delta_dis(batch_ref, batch1, batch2)
            gt_dis = np.sum(
                (
                    np.square(ref_gt_train[idx] - x1_gt_train[idx])
                    - np.square(ref_gt_train[idx] - x2_gt_train[idx])
                ),
                -1,
            )
            gt = torch.tensor(gt_dis > 0, dtype=torch.float32) * 2 - 1

            loss = loss_fn(gt, delta_dis)
            loss.backward()
            optimizer.step()

        # Evaluate.
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            idx = np.arange(n_val)
            batch_ref = torch.tensor(ref_val[idx], dtype=torch.float32)
            batch1 = torch.tensor(x1_val[idx], dtype=torch.float32)
            batch2 = torch.tensor(x2_val[idx], dtype=torch.float32)
            delta_dis = model.triplet_delta_dis(batch_ref, batch1, batch2)
            pred = delta_dis > 0
            gt_dis = np.sum(
                (
                    np.square(ref_gt_val[idx] - x1_gt_val[idx])
                    - np.square(ref_gt_val[idx] - x2_gt_val[idx])
                ),
                -1,
            )
            gt = torch.tensor(gt_dis > 0)
            n_correct += (pred == gt).sum().item()
            n_total += len(idx)

        acc = n_correct / n_total
        val_acc.append(acc)

        if epoch > 10 and np.mean(val_acc[-10:]) < np.mean(val_acc[-11:-1]):
            break

    print(
        f"{np.round(time.time()- t, 1)}s ({epoch} epochs) | DisEmbed (n={n_pref_data}) fitted with val acc.: {acc}"
    )

    return model, acc


class AE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=latent_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=input_dim),
        )

    def forward(self, x):
        return self.enc(x)

    def reconstruct(self, x):
        return self.dec(self.enc(x))


def fit_ae(inputs, latent_dim=2, batch_size=32, device="cpu"):
    model = AE(input_dim=inputs.shape[1], latent_dim=latent_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    n_data = inputs.shape[0]
    n_train = int(n_data * 0.75)
    n_iter_per_epoch = max(n_train // batch_size, 1)

    epoch = 0
    val_loss = []
    while True:
        for _ in range(n_iter_per_epoch):
            idx = np.random.choice(n_train, batch_size)
            batch = torch.tensor(inputs[idx], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            outputs = model.reconstruct(batch)
            loss = loss_fn(outputs, batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            idx = np.arange(n_train, n_data)
            batch = torch.tensor(inputs[idx], dtype=torch.float32).to(device)
            outputs = model.reconstruct(batch)
            val_loss.append(loss_fn(outputs, batch).item())

        epoch += 1
        if epoch > 10 and np.mean(val_loss[-10:]) > np.mean(val_loss[-11:-1]):
            break

    print(
        f"{epoch} epochs | AE fitted with reconstruction loss: {np.mean(val_loss[-10:])}"
    )
    return model


def fit_pca(features, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(features)
    return pca
