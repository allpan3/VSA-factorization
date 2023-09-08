import torch
import torch.utils.data as data
from vsa import VSA
import os.path

class VSADataset(data.Dataset):
    def __init__(self, root, num_samples, vsa: VSA, algo, num_vectors_superposed=1, noise=0.0):
        super(VSADataset, self).__init__()

        # Algo 1 and 2 samples are the same, but for convenience we'll generate separate copies
        sample_file = os.path.join(root, f"samples-{algo}-{num_vectors_superposed}s-{noise}n-{num_samples}.pt")

        if (os.path.exists(sample_file)):
            self.labels, self.data = torch.load(sample_file)
        else:   
            if algo == "ALGO3":
                # Labels do not contain ID
                self.labels, objects = vsa.sample(num_samples, num_factors=vsa.num_factors-1, num_vectors_superposed=num_vectors_superposed, noise=noise)
                # Final codebook is ID. For n vectors superposed, only need to bundle the firt n IDs
                ids = vsa.multiset(vsa.codebooks[-1][0:num_vectors_superposed])
                self.data = vsa.bind(objects, ids)
            else:
                self.labels, self.data = vsa.sample(num_samples,num_vectors_superposed=num_vectors_superposed, noise=noise)
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
