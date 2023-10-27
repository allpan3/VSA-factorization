import torch
import torch.utils.data as data
import os.path
import itertools
import random
from torch import Tensor
import random as rand
from vsa import VSA

class VSADataset(data.Dataset):
    def __init__(self, root, num_samples, vsa: VSA, algo, num_vectors_superposed: int or list or range = 1, quantize=False, noise=0.0, device=None):
        super(VSADataset, self).__init__()

        self.vsa = vsa

        # Turn num_vectors_superposed into a list if it is an integer, for easier processing
        if (type(num_vectors_superposed) == int):
            num_vectors_superposed = [num_vectors_superposed]
        
        num_samples = num_samples // len(num_vectors_superposed)

        sample_files = self._get_filename(root, num_vectors_superposed, algo, quantize, noise, num_samples)

        self.labels = []
        self.data = []
        if (self._check_exists(sample_files)):
            for file in sample_files:
                labels, data = torch.load(file, map_location=device)
                self.labels += labels
                self.data += data
        else:
            # When the number of vectors superposed is a single number 1, we want to directly call resonator network and skip algorithm (cuz algos are for multi-vector extraction)
            for n in num_vectors_superposed:
                if algo == "ALGO3":
                    # Sample vectors without the ID, do not superpose (bundle) them yet
                    labels, vectors = self.sample(num_samples, num_factors=vsa.num_factors-1, num_vectors=n, quantize=quantize, bundled=False, noise=noise)
                    for i in range(num_samples):
                        vectors[i] = self.lookup_algo3(labels[i], vectors[i])
                    data = torch.stack(vectors)
                else:
                    labels, data = self.sample(num_samples,num_vectors=n, quantize=quantize, bundled=True, noise=noise)

                torch.save((labels, data), self._get_filename(root, n, algo, quantize, noise, num_samples))
                self.labels += labels
                self.data += data

        

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

    def _check_exists(self, filenames: list) -> bool:
        return all(os.path.exists(file) for file in filenames)

    def _get_filename(self, root, num_vectors_superposed, algo, quantize, noise, num_samples) -> str or list(str):
        # Algo 3 samples are different from the rest, so we'll generate them separately (when num_vectors_superposed is 1 they are actually the same but we ignore for convenience)
        if (type(num_vectors_superposed) == int):
            if algo == "ALGO3":
                return os.path.join(root, f"samples-{num_vectors_superposed}obj-algo3-{self._name_quantized(quantize)}-{noise}n-{num_samples}.pt")
            else:
                return os.path.join(root, f"samples-{num_vectors_superposed}obj-{self._name_quantized(quantize)}-{noise}n-{num_samples}.pt")
        else:
            if algo == "ALGO3":
                return [os.path.join(root, f"samples-{n}obj-algo3-{self._name_quantized(quantize)}-{noise}n-{num_samples}.pt") for n in num_vectors_superposed]
            else:
                return [os.path.join(root, f"samples-{n}obj-{self._name_quantized(quantize)}-{noise}n-{num_samples}.pt") for n in num_vectors_superposed]

    def _name_quantized(self, quantized: bool):
        return "quantized" if quantized else "expanded"