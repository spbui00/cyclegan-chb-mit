from data.base_dataset import BaseDataset
import numpy as np
import torch


class DummyDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_samples',    type=int,   default=2000, help='number of windows to generate')
        parser.add_argument('--window_size',  type=int,   default=256,  help='length of each time-series window')
        parser.add_argument('--noise_std',    type=float, default=0.5,  help='std dev of Gaussian noise')
        parser.set_defaults(max_dataset_size=2000)
        return parser
    def __init__(self, opt):
        super().__init__(opt)
        self.n_samples   = opt.n_samples
        self.window_size = opt.window_size
        self.noise_std   = opt.noise_std
        # dummy transform placeholder (unused for numeric data)

        self.data_A = []
        self.data_B = []
        for idx in range(self.n_samples):
            rng   = np.random.RandomState(idx)
            freq  = rng.uniform(1.0, 10.0)
            phase = rng.uniform(0.0, 2 * np.pi)
            t     = np.linspace(0, 1, self.window_size, endpoint=False, dtype=np.float32)
            clean = np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
            noise = rng.normal(0, self.noise_std, self.window_size).astype(np.float32)
            noisy = clean + noise
            self.data_A.append(clean)
            self.data_B.append(noisy)

    def __getitem__(self, index):
        clean = self.data_A[index]
        noisy = self.data_B[index]

        # convert to tensors with channel dim => [1, L]
        data_A = torch.from_numpy(clean).unsqueeze(0)
        data_B = torch.from_numpy(noisy).unsqueeze(0)
        return {
            'data_A': data_A,
            'data_B': data_B,
            'paths':   str(index)
        }

    def __len__(self):
        """Return the total number of images."""
        return self.n_samples
