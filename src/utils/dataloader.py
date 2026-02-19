import numpy as np
from torch.utils.data import DataLoader

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
