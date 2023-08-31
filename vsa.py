# %%
import torchhd as hd
import torch
from torchhd.types import VSAOptions
import itertools
from torchvision.datasets import utils
import os.path
from typing import List, Tuple
import random
import pickle

# %%
class VSA:
    # Dictionary of all vectors composed from factors
    dict = {}
    # codebooks for each factor
    codebooks: List[hd.VSATensor] or hd.VSATensor

    def __init__(
            self,
            root: str,
            dim: int,
            model:VSAOptions,
            num_factors: int,
            num_codevectors: int or Tuple[int], # number of vectors per factor, or tuple of number of codevectors for each factor
            device = "cpu"
        ):
        super(VSA, self).__init__()

        self.root = root
        self.model = model
        self.device = device
        # # MAP default is float, we want to use int
        # if model == 'MAP':
        #     self.dtype = torch.int8
        # else:
        self.dtype = None
        self.dim = dim
        self.num_factors = num_factors
        self.num_codevectors = num_codevectors

        if self._check_exists("codebooks.pt"):
            self.codebooks = torch.load(os.path.join(self.root, "codebooks.pt"))
        else:
            self.codebooks = self.gen_codebooks()

        if self._check_exists("dict.pkl"):
            self.dict = pickle.load(open(os.path.join(self.root, "dict.pkl"), "rb"))
        else:
            self.dict = self.gen_dict()


    def gen_codebooks(self) -> List[hd.VSATensor] or hd.VSATensor:
        l = []
        # All factors have the same number of vectors
        if (type(self.num_codevectors) == int):
            for i in range(self.num_factors):
                l.append(hd.random(self.num_codevectors, self.dim, vsa=self.model, dtype=self.dtype, device=self.device))
            l = torch.stack(l).to(self.device)
        # Every factor has a different number of vectors
        else:
            for i in range(self.num_factors):
                l.append(hd.random(self.num_codevectors[i], self.dim, vsa=self.model, dtype=self.dtype, device=self.device))
            
        os.makedirs(self.root, exist_ok=True)
        torch.save(l, os.path.join(self.root, f"codebooks.pt"))

        return l


    def gen_dict(self):
        '''
        Generate dictionary of all possible combinations of factors
        key is a tuple of indices of each factor
        value is the tensor of the compositional vector
        '''
        d = {}
        for key in itertools.product(*[range(len(self.codebooks[i])) for i in range(self.num_factors)]):
            d[key] = hd.multibind(torch.stack([self.codebooks[j][key[j]] for j in range(self.num_factors)]))
        
        pickle.dump(d, open(os.path.join(self.root, "dict.pkl"), "wb"))
        return d
        
    
    def sample(self, num_samples, num_vectors_supoerposed = 1, noise=0.0):
        '''
        Sample `num_samples` random vectors from the dictionary, or multiple vectors superposed
        '''
        labels = [None] * num_samples
        vectors = hd.empty(num_samples, self.dim, vsa=self.model, dtype=self.dtype, device=self.device)
        for i in range(num_samples):
            labels[i]= [tuple([random.randint(0, len(self.codebooks[i])-1) for i in range(self.num_factors)]) for j in range(num_vectors_supoerposed)]
            vectors[i] = self.apply_noise(self.__getitem__(labels[i]), noise)
        return labels, vectors

    def apply_noise(self, vector, noise):
        # orig = vector.clone()
        indices = [random.random() < noise for i in range(self.dim)]
        vector[indices] = self.flip(vector[indices])
        
        # print("Verify noise:" + str(hd.dot_similarity(orig, vector)))
        return vector.to(self.device)
    
    def flip(self, vector):
        if (self.model == 'MAP'):
            return -vector
        elif (self.model == "BSC"):
            return 1 - vector

    def cleanup(self, input):
        '''
        input: `(n, d)` :tensor or [(d): tensor] * n :list
        n must match the number of factors
        '''
        # assert(len(input) == self.num_factors)
        indices = [None] * self.num_factors
        if self.model == 'MAP':
            for i in range(self.num_factors):
                winner = torch.argmax(torch.abs(hd.dot_similarity(input[i], self.codebooks[i])))
                # winner = torch.argmax(hd.dot_similarity(input[i], self.codebooks[i]))
                indices[i] = winner.item()
        # elif self.model == "BSC":

        return tuple(indices)

    def similarity(self, input: hd.VSATensor, others: hd.VSATensor) -> hd.VSATensor:
        """Inner product with other hypervectors.
        Shapes:
            - input: :math:`(*, d)`
            - others: :math:`(n, d)` or :math:`(d)`
            - output: :math:`(*, n)` or :math:`(*)`, depends on shape of others
        """
        if isinstance(input, hd.MAPTensor):
            return input.dot_similarity(others)
        elif isinstance(input, hd.BSCTensor):
            return hd.hamming_similarity(input, others)


    def ensure_vsa_tensor(self, data):
        return hd.ensure_vsa_tensor(data, vsa=self.model, dtype=self.dtype, device=self.device)

    def __getitem__(self, key: list):
        '''
        `key` is a list of tuples in [(f0, f1, f2, ...), ...] format.
        fx is the index of the factor in the codebook, which is also its label.
        '''
        if (len(key) == 1):
            return self.dict[key[0]].to(self.device)
        else:
            obj = self.dict[key[0]].to(self.device)
            i = 1
            while i < len(key):
                obj = hd.bundle(obj, self.dict[key[i]].to(self.device))
                i += 1
            return obj
    
 
    def _check_exists(self, file) -> bool:
        return utils.check_integrity(os.path.join(self.root, file))
# %%
