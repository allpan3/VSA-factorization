import torch.nn as nn
import torchhd as hd
from typing import Literal
import torch
from vsa import VSA

class Resonator(nn.Module):
    def __init__(self, vsa:VSA, codebooks, norm=False, activation='NONE', iterations=100):
        super(Resonator, self).__init__()
        self.vsa = vsa
        self.codebooks = codebooks
        init_estimates = hd.multiset(codebooks)

        if norm:
            init_estimates = self.normalize(init_estimates)
        self.norm = norm
        self.init_estimates = init_estimates
        self.iterations = iterations
        self.activation = activation

    def forward(self, input):
        return self.resonator_network(input, self.init_estimates, self.codebooks, self.iterations, self.norm, self.activation)

    def resonator_network(self, input, estimates, codebooks, iterations, norm, activation):
        old_estimates = estimates
        if norm:
            input = self.normalize(input)
        for k in range(iterations):
            estimates = self.resonator_stage(input, estimates, codebooks, activation=activation)
            if all((estimates == old_estimates).flatten().tolist()):
                break
            old_estimates = estimates

        # outcome: the indices of the codevectors in the codebooks
        outcome = self.vsa.cleanup(estimates)

        return outcome, k 


    def resonator_stage(self,
                        input: hd.VSATensor,
                        estimates: hd.VSATensor or list,
                        codebooks: hd.VSATensor or list,
                        activation: Literal['NONE', 'ABS', 'ZERO'] = 'NONE'):
        if type(estimates) is list:
            pass
        else:
            n = estimates.size(-2)

            # Get binding inverse of the estimates
            inv_estimates = estimates.inverse()

            # Roll over the number of estimates to align each row with the other symbols
            # Example: for factorizing x, y, z the stacked matrix has the following estimates:
            # [[z, y],
            #  [x, z],
            #  [y, x]]
            rolled = []
            for i in range(1, n):
                rolled.append(inv_estimates.roll(i, -2))

            inv_estimates = torch.stack(rolled, dim=-2)

            # First bind all the other estimates together: z * y, x * z, y * z
            inv_others = hd.multibind(inv_estimates)

            # Then unbind all other estimates from the input: s * (x * y), s * (x * z), s * (y * z)
            new_estimates = hd.bind(input.unsqueeze(-2), inv_others)

            similarity = hd.dot_similarity(new_estimates.unsqueeze(-2), codebooks)

            if (activation == 'ABS'):
                similarity = torch.abs(similarity)

            output = hd.dot_similarity(similarity, codebooks.transpose(-2, -1)).squeeze(-2)
            # This should be normalizing back to 1 and -1, but sign can potentially keep 0 at 0. It's very unlikely to see a 0 and sign() is fast 
            output = output.sign()
            
            return output

    
    def normalize(self, input):
        if isinstance(input, hd.MAPTensor):
            return hd.hard_quantize(input)
        elif isinstance(input, hd.BSCTensor):
            # positive = torch.tensor(True, dtype=input.dtype, device=input.device)
            # negative = torch.tensor(False, dtype=input.dtype, device=input.device)
            # return torch.where(input >= 0, positive, negative)

            # Seems like BSCTensor is automatically normalized after bundle
            return input