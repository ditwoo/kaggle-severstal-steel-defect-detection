import numpy as np
from .steel import SteelDataset
from torch.utils.data import Sampler


class BalancedSampler(Sampler):
    def __init__(self, dataset: SteelDataset):
        super().__init__(dataset)
        rles = dataset.rles
        # find rows with present mask by channel
        all_missed = []
        chnls = [[], [], [], []]
        for idx, rle in enumerate(rles):
            # check if all channels missing
            if all(len(rle[c]) == 0 for c in range(4)):
                all_missed.append(idx)
            else:
                for c in range(4):
                    if len(rle[c]) != 0:
                        chnls[c].append(idx)
        
        self.neg_idx = np.array(all_missed)
        self.c1_idx = np.array(chnls[0])
        self.c2_idx = np.array(chnls[1])
        self.c3_idx = np.array(chnls[2])
        self.c4_idx = np.array(chnls[3])
    
        self.num_images = len(dataset) // 4
        self.length = self.num_images * 5

    def __len__(self):
        return self.length

    def __iter__(self):
        # generate indices to use for iteration
        neg = np.random.choice(self.neg_idx, self.num_images, replace=True)
        c1 = np.random.choice(self.c1_idx, self.num_images, replace=True)
        c2 = np.random.choice(self.c2_idx, self.num_images, replace=True)
        c3 = np.random.choice(self.c3_idx, self.num_images, replace=True)
        c4 = np.random.choice(self.c4_idx, self.num_images, replace=True)

        l = np.stack([neg, c1, c2, c3, c4]).T
        l = l.reshape(-1)
        return iter(l)
