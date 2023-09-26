import torch
import torch.utils.data as data
import os.path
import itertools
import random
from torch import Tensor
import random as rand
from vsa import VSA

class VSADataset(data.Dataset):
    def __init__(self, root, num_samples, vsa: VSA, algo, num_vectors_superposed=1, quantize=False, noise=0.0):
        super(VSADataset, self).__init__()

        self.vsa = vsa

        # Algo 1 and 2 samples are the same, but for convenience we'll generate separate copies
        sample_file = os.path.join(root, f"samples-{algo}-{num_vectors_superposed}s-{quantize}q-{noise}n-{num_samples}.pt")

        if (os.path.exists(sample_file)):
            self.labels, self.data = torch.load(sample_file)
        else:
            if num_vectors_superposed != 1 and algo == "ALGO3":
                # Sample vectors without the ID, do not superpose (bundle) them yet
                self.labels, vectors = self.sample(num_samples, num_factors=vsa.num_factors-1, num_vectors=num_vectors_superposed, quantize=quantize, bundled=False, noise=noise)
                for i in range(num_samples):
                    vectors[i] = self.lookup_algo3(self.labels[i], vectors[i])
                self.data = torch.stack(vectors)
            else:
                self.labels, self.data = self.sample(num_samples,num_vectors=num_vectors_superposed, quantize=quantize, bundled=True, noise=noise)
            torch.save((self.labels, self.data), sample_file)

    def sample(self, num_samples, num_factors = None, num_vectors = 1, quantize = False, bundled = True, noise=0):
        '''
        Generate `num_samples` random samples, each containing `num_vectors` compositional vectors.
        If `bundled` is True, these vectors are bundled into one unquantized vector, else a list of `num_vectors` quantized vectors are returned.
        The vector is composed of factors from the first n `num_factors` codebooks if `num_factors` is specified. Otherwise it is composed of all available factors
        When `bundled` is True, `quantize` controls whether the sampled vector is quantized or not (even if `num_vectors` is 1).
        When `bundled` is False, the individual vectors are always quantized.
        '''

        if num_factors == None:
            num_factors = self.vsa.num_factors

        assert(num_factors <= self.vsa.num_factors)

        labels = [None] * num_samples
        vectors = [[] for _ in range(num_samples)]
        for i in range(num_samples):
            labels[i] = [tuple([random.randint(0, len(self.vsa.codebooks[i])-1) for i in range(num_factors)]) for j in range(num_vectors)]
            if bundled:
                vectors[i] = self.vsa.get_vector(labels[i], quantize=quantize) if noise == 0 else self.vsa.apply_noise(self.vsa.get_vector(labels[i], quantize=quantize), noise, quantize)
            else:
                # Intentionally not stacked since we want to keep the vectors separate
                vectors[i] = [self.vsa.get_vector(labels[i][j], quantize=True) if noise == 0 else self.vsa.apply_noise(self.vsa.get_vector(labels[i][j], quantize=True), noise, True) for j in range(num_vectors)]
        try: 
            vectors = torch.stack(vectors)
        except:
            pass
        return labels, vectors


    def lookup_algo3(self, label, vectors=None, bundled=True):
        if vectors is None:
            vectors = [self.vsa.get_vector(label[i], quantize=True) for i in range(len(label))]

        rule = [x for x in itertools.product(range(len(self.vsa.codebooks[0])), range(len(self.vsa.codebooks[1])))]
        # Reorder the positions of the vectors in each label in the ascending order of the first 2 factors
        _, vectors = list(zip(*sorted(zip(label, vectors), key=lambda k: rule.index(k[0][0:2]))))
        # Remember the original indice of the codebooks for reordering later
        indices = sorted(range(len(label)), key=lambda k: rule.index(label[k][0:2]))
        # Bind the vector with ID determined by the position in the list
        vectors = [self.vsa.bind(vectors[j], self.vsa.codebooks[-1][j]) for j in range(len(label))]
        # Return to the original order (for similarity check)
        vectors = [vectors[i] for i in indices]
        if bundled:
            vectors = VSA.multiset(torch.stack(vectors), quantize=True)
        return vectors




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
