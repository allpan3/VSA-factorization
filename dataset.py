import torch
import torch.utils.data as data
from vsa import VSA
import os.path

class VSADataset(data.Dataset):
    def __init__(self, root, num_samples, vsa: VSA, num_vectors_superposed=1, noise=0.0):
        super(VSADataset, self).__init__()

        self.root = root
        self.num_samples = num_samples
        self.vsa = vsa
        self.num_vectors_superposed = num_vectors_superposed
        self.noise = noise


        sample_file = os.path.join(root, f"samples-{num_samples}s-{noise}n.pt")

        if (os.path.exists(sample_file)):
            self.labels, self.data = torch.load(sample_file)
        else:   
            self.labels, self.data = self.vsa.sample(num_samples, num_vectors_supoerposed=1, noise=noise)
            torch.save((self.labels, self.data), sample_file)

    def __getitem__(self, index: int):
        '''
        Args:
            index (int): Index
        
        Returns:
            tuple: (data, label)
        '''
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.data)
