from torch.utils.data import Dataset
import numpy as np


class VF_FM(Dataset):
    def __init__(self, data, all_vel=True) -> None:
        super().__init__()
        self.all_vel = all_vel
        self._preprocess(data, 'data')

        if self.data.ndim == 4:
            self.shape = self.data.shape[1:]
            self.one_yp = True
            self.wm_vf = False
        elif self.data.ndim == 5:
            if self.all_vel == True:
                self.wm_vf = False
                self.shape = self.data.shape[2:]
                self.num_yp = self.data.shape[1]
            else:
                self.wm_vf = True
            self.one_yp = False
        else:
            raise ValueError("Check the members of the dataset!")

    def _preprocess(self, data, name):
        setattr(self, name, (data).astype(np.float32))

    def __len__(self):
        if self.one_yp or self.wm_vf:
            return self.data.shape[0]
        else:
            return self.data.shape[0]*self.data.shape[1]

    def __getitem__(self, index):
        if self.one_yp and not self.wm_vf:
            return  np.empty(self.shape, dtype=np.float32), self.data[index]

        elif not self.one_yp and not self.wm_vf:
            yp_ind = index % self.num_yp
            batch = index // self.num_yp
            return np.empty(self.shape, dtype=np.float32), self.data[batch, yp_ind], yp_ind

        else:
            return self.data[index, 0], self.data[index, 1]


DATASETS = {"VF_FM": VF_FM}
