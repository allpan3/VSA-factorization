############################################################################################
# The VSA operation portion is inspired by torchhd library, tailored for our project purpose
# We use only the MAP model, but with two variants: software and hardware models
# In the software model, the definitions and operations are exactly the same as MAP model
# In the hardware model, the representation is BSC-like, but the operations are still MAP-like.
# We use binary in place of bipolar whenever possible to reduce the complexity of the hardware,
# but the bahaviors still closely follow MAP model.
# For example, we use XNOR for binding to get the exact same input-output mapping. We bipolize the
# values in bundle and hamming distance operations to get the same results as MAP model.
# The original library is located at:
# https://github.com/hyperdimensional-computing/torchhd.git
############################################################################################

# %%
import torch
from torch import Tensor
import itertools
from torchvision.datasets import utils
import os.path
from typing import List, Tuple
import random
from typing import Set, Literal

# %%
class VSA:
    # codebooks for each factor
    codebooks: List[Tensor] or Tensor

    def __init__(
            self,
            root: str,
            model: Literal['SOFTWARE', 'HARDWARE'],
            dim: int,
            num_factors: int,
            num_codevectors: int or Tuple[int], # number of vectors per factor, or tuple of number of codevectors for each factor
            seed: None or int = None,  # random seed
            device = "cpu"
        ):

        self.root = root
        self.model = model
        self.device = device
        # default is float, we may want to use int
        self.dtype = torch.int8
        self.dim = dim
        self.num_factors = num_factors
        self.num_codevectors = num_codevectors

        # Assign functions
        if (model == "SOFTWARE"):
            self.random = self._random_software
            self.similarity = self._similarity_software
            self.bind = self._bind_software
            self.multibind = self._multibind_software
            self.bundle = self._bundle_software
            self.multiset = self._multiset_software
        elif (model == "HARDWARE"):
            self.random = self._random_hardware
            self.similarity = self._similarity_hardware
            self.bind = self._bind_hardware
            self.multibind = self._multibind_hardware
            self.bundle = self._bundle_hardware
            self.multiset = self._multiset_hardware

        # Generate codebooks
        if self._check_exists("codebooks.pt"):
            self.codebooks = torch.load(os.path.join(self.root, "codebooks.pt"), map_location=self.device)
        else:
            self.codebooks = self.gen_codebooks(seed)


    def gen_codebooks(self, seed) -> List[Tensor] or Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        l = []
        # All factors have the same number of vectors
        if (type(self.num_codevectors) == int):
            for i in range(self.num_factors):
                l.append(self.random(self.num_codevectors, self.dim))
            l = torch.stack(l).to(self.device)
        # Every factor has a different number of vectors
        else:
            for i in range(self.num_factors):
                l.append(self.random(self.num_codevectors[i], self.dim))

        os.makedirs(self.root, exist_ok=True)
        torch.save(l, os.path.join(self.root, f"codebooks.pt"))

        return l

    def sample(self, num_samples, num_vectors_supoerposed = 1, noise=0.0):
        '''
        Sample `num_samples` random vectors from the dictionary, or multiple vectors superposed
        '''
        labels = [None] * num_samples
        vectors = self.empty(num_samples, self.dim)
        for i in range(num_samples):
            labels[i]= [tuple([random.randint(0, len(self.codebooks[i])-1) for i in range(self.num_factors)]) for j in range(num_vectors_supoerposed)]
            vectors[i] = self.apply_noise(self.__getitem__(labels[i]), noise)
        return labels, vectors

    def apply_noise(self, vector, noise):
        # orig = vector.clone()
        indices = [random.random() < noise for i in range(self.dim)]
        def flip(vector):
            if self.model == "SOFTWARE":
                return -vector
            elif self.model == "HARDWARE":
                return 1 - vector

        vector[indices] = flip(vector[indices])
        
        # print("Verify noise:" + str(self.similarity(orig, vector)))
        return vector.to(self.device)

    def cleanup(self, inputs, abs):
        '''
        input: `(b, f, d)` :tensor. b is batch size, f is number of factors, d is dimension
        Return: List[Tuple(int)] of length b
        '''
        if type(self.codebooks) == list:
            winners = torch.empty((inputs.size(0), self.num_factors), dtype=torch.int8, device=self.device)
            for i in range(self.num_factors):
                if abs:
                    winners[:,i] = torch.argmax(torch.abs(self.similarity(inputs[:,i], self.codebooks[i])), -1)
                else:
                    winners[:,i] = torch.argmax(self.similarity(inputs[:,i], self.codebooks[i]), -1)
            return [tuple(winners[i].tolist()) for i in range(winners.size(0))]
        else:
            if abs:
                winners = torch.argmax(torch.abs(self.similarity(inputs.unsqueeze(-2), self.codebooks).squeeze(-2)), -1)
            else:
                winners = torch.argmax(self.similarity(inputs.unsqueeze(-2), self.codebooks).squeeze(-2), -1)
            return [tuple(winners[i].tolist()) for i in range(winners.size(0))]
      

    def get_vector(self, key:tuple):
        '''
        `key` is a tuple of indices of each factor
        Instead of pre-generate the dictionary, we combine factors to get the vector on the fly
        This saves meomry, and also the dictionary lookup is only used during sampling and comparison
        '''
        assert(len(key) == self.num_factors)
        factors = [self.codebooks[i][key[i]] for i in range(self.num_factors)]
        return self.multibind(torch.stack(factors)).to(self.device)

    def __getitem__(self, key: list):
        '''
        `key` is a list of tuples in [(f0, f1, f2, ...), ...] format.
        fx is the index of the factor in the codebook, which is also its label.
        '''
        if (len(key) == 1):
            return self.get_vector(key[0])
        else:
            # TODO to be tested
            return self.multiset(torch.stack([self.get_vector(key[i]) for i in range(len(key))]))

    def _check_exists(self, file) -> bool:
        return utils.check_integrity(os.path.join(self.root, file))

    def empty(self, num_vectors: int, dimensions: int) -> Tensor:
        return torch.empty(num_vectors, dimensions, dtype=self.dtype, device=self.device)

    def _random_software(self, num_vectors: int, dimensions: int) -> Tensor:
        size = (num_vectors, dimensions)
        select = torch.empty(size, dtype=torch.bool, device=self.device)
        select.bernoulli_(generator=None)

        result = torch.where(select, -1, +1).to(dtype=self.dtype, device=self.device)
        return result

    def _random_hardware(self, num_vectors: int, dimensions: int) -> Tensor:
        size = (num_vectors, dimensions)
        select = torch.empty(size, dtype=torch.bool, device=self.device)
        select.bernoulli_(generator=None)

        result = torch.where(select, 0, 1).to(dtype=self.dtype, device=self.device)
        return result

    def _bind_software(self, input: Tensor, others: Tensor) -> Tensor:
        return torch.mul(input, others)

    def _bind_hardware(self, input: Tensor, others: Tensor) -> Tensor:
        """XNOR"""
        return torch.logical_not(torch.logical_xor(input, others)).to(self.dtype)

    def _multibind_software(self, inputs: Tensor) -> Tensor:
        """Bind multiple hypervectors"""
        if inputs.dim() < 2:
            raise RuntimeError(
                f"data needs to have at least two dimensions for multibind, got size: {tuple(inputs.shape)}"
            )
        return torch.prod(inputs, dim=-2, dtype=inputs.dtype)

    def _multibind_hardware(self, inputs: Tensor) -> Tensor:
        if inputs.dim() < 2:
            raise RuntimeError(
                f"data needs to have at least two dimensions for multibind, got size: {tuple(inputs.shape)}"
            )
        
        def biggest_power_two(n):
            """Returns the biggest power of two <= n"""
            # if n is a power of two simply return it
            if not (n & (n - 1)):
                return n

            # else set only the most significant bit
            return int("1" + (len(bin(n)) - 3) * "0", 2)

        n = inputs.size(-2)
        n_ = biggest_power_two(n)
        output = inputs[..., :n_, :]

        # parallelize many XORs in a hierarchical manner
        # for larger batches this is significantly faster
        while output.size(-2) > 1:
            output = self.bind(output[..., 0::2, :], output[..., 1::2, :])

        output = output.squeeze(-2)

        # TODO: as an optimization we could also perform the hierarchical XOR
        # on the leftovers in a recursive fashion
        leftovers = torch.unbind(inputs[..., n_:, :], -2)
        for i in range(n - n_):
            output = self.bind(output, leftovers[i])

        return output.to(inputs.dtype)

    def _bundle_software(self, input: Tensor, others: Tensor) -> Tensor:
        return torch.add(input, others) 

    def _bundle_hardware(self, input: Tensor, others: Tensor, normalize = True) -> Tensor:
        """
        Bipolarize the values, then add them up.
        """
        min_one = torch.tensor(-1, dtype=self.dtype, device=self.device)
        input_as_bipolar = torch.where(input == 0, min_one, input)
        others_as_bipolar = torch.where(others == 0, min_one, others)

        result = torch.add(input_as_bipolar, others_as_bipolar)
        # Binarize. Tie goes to 1 (take the sign bit)
        if normalize:
            result = torch.where(result < 0, 0, 1)
        return result

    def _multiset_software(self, inputs: Tensor, weights: Tensor = None, normalize=False) -> Tensor:
        """Bundle multiple hypervectors"""
        if inputs.dim() < 2:
            raise RuntimeError(
                f"data needs to have at least two dimensions for multiset, got size: {tuple(inputs.shape)}"
            )
        # One weight for each vector in inputs
        if weights != None:
            assert(inputs.size(-2) == weights.size(-1))
            result = torch.matmul(weights.type(torch.float32), inputs.type(torch.float32))
        else:
            result = torch.sum(inputs, dim=-2, dtype=torch.float32)
        
        if normalize:
            result = torch.where(result < 0, -1, 1).type(inputs.dtype)
        
        return result
    
    def _multiset_hardware(self, inputs: Tensor, weights: Tensor = None, normalize = True) -> Tensor:
        min_one = torch.tensor(-1, dtype=inputs.dtype, device=inputs.device)
        inputs_as_bipolar = torch.where(inputs == 0, min_one, inputs) 
        if weights != None:
            assert(inputs.size(-2) == weights.size(-1))
            result = torch.matmul(weights.type(torch.float32), inputs_as_bipolar.type(torch.float32)) 
        else:
            result = torch.sum(inputs_as_bipolar, dim=-2, dtype=torch.float32)

        if normalize:
            result = torch.where(result < 0, 0, 1).type(inputs.dtype)

        return result


    def _similarity_software(self, input: Tensor, others: Tensor) -> Tensor:
        """Inner product between hypervectors.
        Shapes:
            - input: :math:`(*, d)`
            - others: :math:`(n, d)` or :math:`(d)`
            - output: :math:`(*, n)` or :math:`(*)`, depends on shape of others
        """
        if others.dim() >= 2:
            others = others.transpose(-2, -1)
        return torch.matmul(input.type(torch.float32), others.type(torch.float32))

    def _similarity_hardware(self, input: Tensor, others: Tensor) -> Tensor:
        '''Hamming similarity-like implementation except add -1 when unequal.
           The result is exactly the same as dot product. Since vectors are expected
           to be binary, matmul is not required.
        Shapes:
            - input: :math:`(*, d)`
            - others: :math:`(n, d)` or :math:`(d)`
            - output: :math:`(*, n)` or :math:`(*)`, depends on shape of others
        '''
        if input.dim() > 1 and others.dim() > 1:
            bipolar = torch.where(input.unsqueeze(-2) == others.unsqueeze(-3), 1, -1)
        else:
            bipolar = torch.where(input == others, 1, -1)

        return torch.sum(bipolar, dim=-1, dtype=torch.float32)

    def normalize(self, input):
        if self.model == "SOFTWARE":
            positive = torch.tensor(1, dtype=input.dtype, device=input.device)
            negative = torch.tensor(-1, dtype=input.dtype, device=input.device)
        elif self.model == "HARDWARE":
            positive = torch.tensor(1, dtype=input.dtype, device=input.device)
            negative = torch.tensor(0, dtype=input.dtype, device=input.device)
        return torch.where(input >= 0, positive, negative)