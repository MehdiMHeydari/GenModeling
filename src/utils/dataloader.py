import os
import numpy as np
from torch.utils.data import DataLoader


def get_data_loader(data_path, batch_size, dataset_cls, train_samples=9000,
                    save_dir=None, loader_type="darcy"):
    """Generic data loader dispatcher.

    Args:
        loader_type: "darcy", "ns", or "cns" (compressible NS). Default "darcy".
        Other args: passed to the specific loader.
    Returns:
        train_loader, data_min, data_max
        (Scalars, used by the training scripts' log line. For cns they are
        the min/max of the z-scored training data; the per-channel
        mean/std arrays used for denormalization are saved to save_dir as
        data_mean.npy / data_std.npy.)
    """
    if loader_type == "cns":
        return get_cns_loader(data_path, batch_size, dataset_cls,
                              train_samples=train_samples, save_dir=save_dir)
    if loader_type == "ns":
        return get_ns_loader(data_path, batch_size, dataset_cls,
                             train_samples=train_samples, save_dir=save_dir)
    else:
        return get_darcy_loader(data_path, batch_size, dataset_cls,
                                train_samples=train_samples, save_dir=save_dir)


def get_cns_loader(data_path, batch_size, dataset_cls, train_samples=9000,
                   save_dir=None):
    """Load PDEBench 2D Compressible NS with per-channel z-score normalization.

    Expects the preprocessed merged HDF5 (from scripts/preprocess_cns.py)
    with layout (N, 4, 128, 128) under the 'tensor' key. Channels are
    (density, pressure, Vx, Vy).

    Why z-score and not min-max: the channels are heavy-tailed and span
    disparate scales (pressure median 29 / max 557; velocity bulk within
    ±0.5 / single-pixel tails at ±12). Global per-channel min-max is set by
    those outliers and compresses the bulk of the signal into 7-33% of
    [-1, 1] (see diagnostics/cns_mfm_v1/), which starves the small-scale
    channels during training. Per-channel standardization (The Well's
    convention) gives every channel unit variance. Note we deliberately do
    NOT log-transform density/pressure: exp() denormalization would make
    every generated sample trivially positive and destroy the positivity
    physics check.

    Stats are computed on the TRAIN subset only (no test leakage) and saved
    as data_mean.npy / data_std.npy. data_min/max.npy are deliberately not
    written so legacy min-max denorm code fails loudly on CNS.

    Uses the shuffled train indices locked by scripts/lock_cns_test_indices.py
    to avoid the NS temporal-tail OOD issue.
    """
    import h5py

    with h5py.File(data_path, "r") as f:
        data = np.array(f["tensor"], dtype=np.float32)  # (N, C, H, W)

    N, C, H, W = data.shape
    print(f"CNS raw: {data.shape}")

    # Select the train subset FIRST so normalization stats are train-only.
    train_idx_path = "data/cns_train_indices.npy"
    if os.path.exists(train_idx_path):
        train_idx = np.load(train_idx_path)[:train_samples]
        train_data = data[train_idx]
        print(f"CNS: using {len(train_idx)} shuffled train indices from "
              f"{train_idx_path}")
    else:
        train_data = data[:train_samples]
        print(f"CNS: no locked train indices, using first {train_samples} rows")
    del data

    # Per-channel z-score (train-only stats)
    ch_mean = train_data.mean(axis=(0, 2, 3)).astype(np.float32)  # (C,)
    ch_std = train_data.std(axis=(0, 2, 3)).astype(np.float32)    # (C,)
    ch_std = np.maximum(ch_std, 1e-8)
    train_norm = (train_data - ch_mean.reshape(1, C, 1, 1)) \
        / ch_std.reshape(1, C, 1, 1)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "data_mean.npy"), ch_mean)
        np.save(os.path.join(save_dir, "data_std.npy"), ch_std)

    dataset = dataset_cls(train_norm, all_vel=True)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
    )
    print(f"CNS loader: {len(train_norm)} train samples, z-score per channel; "
          f"means={ch_mean.tolist()} stds={ch_std.tolist()}")
    # Returned scalars feed the training scripts' "range [a, b]" log line;
    # here they are the normalized data's actual min/max (z-units).
    return loader, float(train_norm.min()), float(train_norm.max())


def get_darcy_loader(data_path, batch_size, dataset_cls, train_samples=9000,
                     save_dir=None):
    """Load Darcy Flow HDF5 data with min-max normalization to [-1, 1].

    Matches the notebook data pipeline exactly.

    Args:
        data_path: Path to 2D_DarcyFlow_beta1.0_Train.hdf5
        batch_size: Batch size for DataLoader
        dataset_cls: Dataset class (e.g. VF_FM)
        train_samples: Number of samples for training split (rest is val/test)
        save_dir: If provided, saves data_min.npy and data_max.npy here
    Returns:
        train_loader, data_min, data_max
    """
    import h5py

    with h5py.File(data_path, 'r') as f:
        outputs = np.array(f['tensor']).astype(np.float32)

    # Ensure (N, 1, 128, 128)
    if outputs.ndim == 3:
        outputs = outputs[:, np.newaxis, :, :]

    # Min-max normalize to [-1, 1]
    data_min = float(outputs.min())
    data_max = float(outputs.max())
    outputs_norm = 2.0 * (outputs - data_min) / (data_max - data_min) - 1.0

    # Save normalization stats
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "data_min.npy"), np.array(data_min))
        np.save(os.path.join(save_dir, "data_max.npy"), np.array(data_max))

    train_data = outputs_norm[:train_samples]
    dataset = dataset_cls(train_data, all_vel=True)

    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
    )
    return loader, data_min, data_max


def get_ns_loader(data_path, batch_size, dataset_cls, train_samples=90000,
                  save_dir=None):
    """Load PDEBench 2D incompressible Navier-Stokes HDF5 data.

    Expects HDF5 with shape [N_sim, T, H, W, channels] (PDEBench convention).
    Reshapes to [N_sim*T, channels, H, W] and normalizes to [-1, 1].

    Args:
        data_path: Path to NS HDF5 file
        batch_size: Batch size for DataLoader
        dataset_cls: Dataset class (e.g. VF_FM)
        train_samples: Number of samples for training split
        save_dir: If provided, saves data_min.npy and data_max.npy here
    Returns:
        train_loader, data_min, data_max
    """
    import h5py

    with h5py.File(data_path, 'r') as f:
        print(f"NS HDF5 keys: {list(f.keys())}")

        # Try common PDEBench key names
        if 'velocity' in f:
            data = np.array(f['velocity']).astype(np.float32)
        elif 'Vx' in f and 'Vy' in f:
            vx = np.array(f['Vx']).astype(np.float32)
            vy = np.array(f['Vy']).astype(np.float32)
            data = np.stack([vx, vy], axis=-1)
        elif 'tensor' in f:
            data = np.array(f['tensor']).astype(np.float32)
        else:
            raise ValueError(
                f"Cannot find velocity data. Available keys: {list(f.keys())}. "
                "Expected 'velocity', 'Vx'/'Vy', or 'tensor'."
            )

    print(f"Raw NS data shape: {data.shape}")

    # PDEBench convention: [N_sim, T, H, W, V]
    # We want: [N_sim*T, V, H, W] (channels first)
    if data.ndim == 5:
        N_sim, T, H, W, V = data.shape
        # Flatten simulations × timesteps → independent samples
        data = data.reshape(N_sim * T, H, W, V)
        # Channels last → channels first
        data = data.transpose(0, 3, 1, 2)  # [N*T, V, H, W]
    elif data.ndim == 4:
        # Already [N, H, W, V] or [N, V, H, W]
        if data.shape[-1] <= 3:  # likely channels-last
            data = data.transpose(0, 3, 1, 2)
    elif data.ndim == 3:
        # [N, H, W] single channel
        data = data[:, np.newaxis, :, :]

    print(f"Reshaped NS data: {data.shape} (N, C, H, W)")

    # Min-max normalize to [-1, 1] (global across all channels)
    data_min = float(data.min())
    data_max = float(data.max())
    data_norm = 2.0 * (data - data_min) / (data_max - data_min) - 1.0

    # Save normalization stats
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "data_min.npy"), np.array(data_min))
        np.save(os.path.join(save_dir, "data_max.npy"), np.array(data_max))

    # Shuffle before splitting (timesteps from same sim are sequential)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(data_norm))
    data_norm = data_norm[perm]

    train_data = data_norm[:train_samples]
    dataset = dataset_cls(train_data, all_vel=True)

    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
    )
    print(f"NS loader: {len(train_data)} train samples, "
          f"range [{data_min:.4f}, {data_max:.4f}]")
    return loader, data_min, data_max


def _load_data_flat(vf_paths, x_spatial_cutoff, z_spatial_cutoff):
    """Loads data from a flat list of paths."""
    data = [np.load(path) for path in vf_paths]
    data = np.concatenate(data, axis=1)
    if x_spatial_cutoff is not None:
        data = data[:, :, :x_spatial_cutoff]
    if z_spatial_cutoff is not None:
        data = data[..., :z_spatial_cutoff]
    m, s = np.mean(data, axis=(0, 2, 3), keepdims=True), np.std(data, axis=(0, 2, 3), keepdims=True)
    return data, m, s

def _load_data_nested_1d(vf_paths, x_spatial_cutoff, z_spatial_cutoff):
    """Loads data from a 1D nested list of paths."""
    data = []
    for path in vf_paths[0]:
        d = np.load(path)
        if d.ndim == 3:
            d = d[:, None]
        data.append(d)
    data = np.concatenate(data, axis=1)
    if x_spatial_cutoff is not None:
        data = data[:, :, :x_spatial_cutoff]
    if z_spatial_cutoff is not None:
        data = data[..., :z_spatial_cutoff]
    m, s = np.mean(data, axis=(0, 2, 3), keepdims=True), np.std(data, axis=(0, 2, 3), keepdims=True)
    return data, m, s

def _load_data_nested_2d(vf_paths, x_spatial_cutoff, z_spatial_cutoff):
    """Loads data from a 2D nested list of paths."""
    data = [[np.load(path) for path in uvw_path] for uvw_path in vf_paths]
    data = [np.concatenate(uvw, axis=1) for uvw in data]
    data = np.stack(data, axis=1)
    if x_spatial_cutoff is not None:
        data = data[:, :, :, :x_spatial_cutoff]
    if z_spatial_cutoff is not None:
        data = data[..., :z_spatial_cutoff]
    m, s = np.mean(data, axis=(0, 3, 4), keepdims=True), np.std(data, axis=(0, 3, 4), keepdims=True)
    return data, m, s


def get_loaders_vf_fm(vf_paths, batch_size, dataset_, jump=1, all_vel=True, x_spatial_cutoff=None, z_spatial_cutoff=None, time_cutoff=None, patch_dims=None,
                      multi_patch=False, zero_pad=True, distributed=False, num_replicas=None, rank=None):

    def norm(d, m, s):
        return (d-m)/s

    data = []

     # Dispatch to the correct data loading function based on vf_paths structure
    if isinstance(vf_paths[0], str):
        data, m, s = _load_data_flat(vf_paths, x_spatial_cutoff, z_spatial_cutoff)
    elif isinstance(vf_paths[0], list) and isinstance(vf_paths[0][0], str):
        data, m, s = _load_data_nested_1d(vf_paths, x_spatial_cutoff, z_spatial_cutoff)
    elif isinstance(vf_paths[0], list) and isinstance(vf_paths[0][0], list):
        data, m, s = _load_data_nested_2d(vf_paths, x_spatial_cutoff, z_spatial_cutoff)
    else:
        raise ValueError("Unsupported structure for vf_paths")

    data = norm(data, m, s)

    dataset = None
    data_slice = data[:time_cutoff:jump] if time_cutoff is not None else data[::jump]
    if patch_dims is not None:
        dataset = dataset_(data_slice, all_vel, patch_dims, multi_patch, zero_pad)
    else:
        dataset = dataset_(data_slice, all_vel)

    sampler = None
    shuffle = not distributed

    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        assert num_replicas is not None and rank is not None, "num_replicas and rank must be provided for distributed training"
        sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=0, pin_memory=True)
