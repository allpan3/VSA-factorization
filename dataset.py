import torch
import torch.utils.data as data
from vsa import VSA
import os.path
import itertools

class VSADataset(data.Dataset):
    def __init__(self, root, num_samples, vsa: VSA, algo, num_vectors_superposed=1, noise=0.0):
        super(VSADataset, self).__init__()

        self.vsa = vsa

        # Algo 1 and 2 samples are the same, but for convenience we'll generate separate copies
        sample_file = os.path.join(root, f"samples-{algo}-{num_vectors_superposed}s-{noise}n-{num_samples}.pt")

        if (os.path.exists(sample_file)):
            self.labels, self.data = torch.load(sample_file)
        else:   
            if algo == "ALGO3":
                # Sample vectors without the ID, do not superpose (bundle) them yet
                labels, vectors = self.vsa.sample(num_samples, num_factors=vsa.num_factors-1, num_vectors=num_vectors_superposed, bundled=False, noise=noise)
                for i in range(num_samples):
                    labels[i], vectors[i] = self.lookup_algo3(labels[i], vectors[i])
                self.labels = labels
                self.data = torch.stack(vectors)
            else:
                self.labels, self.data = vsa.sample(num_samples,num_vectors=num_vectors_superposed, bundled=True, noise=noise)
            torch.save((self.labels, self.data), sample_file)

    def lookup_algo3(self, label, vectors=None):
        if vectors is None:
            vectors = [self.vsa.get_vector(label[i]) for i in range(len(label))]

        rule = [x for x in itertools.product(range(len(self.vsa.codebooks[0])), range(len(self.vsa.codebooks[1])))]
        # Reorder the positions of the vectors in each label in the ascending order of the first 2 factors
        label_, vectors_ = list(zip(*sorted(zip(label, vectors), key=lambda k: rule.index(k[0][0:2]))))
        label_ = list(label_)  # convert tuple to list
        # Bind the vector with ID determined by the position in the list
        vectors_ = self.vsa.multiset(torch.stack([self.vsa.bind(vectors_[j], self.vsa.codebooks[-1][j]) for j in range(len(label))]))
        return label_, vectors_

        

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
