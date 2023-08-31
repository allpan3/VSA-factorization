# %%
import torch.nn as nn
import torchhd as hd
from typing import Literal, List
import torch
from vsa import VSA

# %%
class Resonator(nn.Module):

    init_estimates: hd.VSATensor

    def __init__(self, vsa:VSA, type="CONCURRENT", norm=False, activation='NONE', iterations=100, device="cpu"):
        super(Resonator, self).__init__()
        self.to(device)

        self.vsa = vsa
        self.device = device
        self.resonator_type = type
        self.norm = norm
        self.init_estimates = self.gen_init_estimates(vsa.codebooks, norm)
        self.iterations = iterations
        self.activation = activation

    def forward(self, input):
        return self.resonator_network(input, self.init_estimates, self.vsa.codebooks, self.iterations, self.norm, self.activation)

    def gen_init_estimates(self, codebooks: hd.VSATensor or List[hd.VSATensor], norm: bool) -> hd.VSATensor:
        if (type(codebooks) == list):
            guesses = [None] * len(codebooks)
            for i in range(len(codebooks)):
                guesses[i] = hd.multiset(codebooks[i])
            init_estimates = torch.stack(guesses)
        else:
            init_estimates = hd.multiset(codebooks)

        if norm:
            init_estimates = self.normalize(init_estimates)
        
        return init_estimates


    def resonator_network(self, input: hd.VSATensor, estimates: hd.VSATensor, codebooks: hd.VSATensor or list, iterations, norm, activation):
        old_estimates = estimates.clone()
        if norm:
            input = self.normalize(input)
        for k in range(iterations):
            if (self.resonator_type == "SEQUENTIAL"):
                estimates = self.resonator_stage_seq(input, estimates, codebooks, activation)
            elif (self.resonator_type == "CONCURRENT"):
                estimates = self.resonator_stage_concur(input, estimates, codebooks, activation)
            if all((estimates == old_estimates).flatten().tolist()):
                break
            old_estimates = estimates.clone()

        # outcome: the indices of the codevectors in the codebooks
        outcome = self.vsa.cleanup(estimates)

        return outcome, k 


    def resonator_stage_seq(self,
                            input: hd.VSATensor,
                            estimates: hd.VSATensor,
                            codebooks: hd.VSATensor or List[hd.VSATensor],
                            activation: Literal['NONE', 'ABS', 'NONNEG'] = 'NONE'):
        n = estimates.size(-2)
        
        for i in range(n):
            # Since we only target MAP and BSC, inverse of a vector itself
            # Remove the currently processing factor itself
            others = hd.multibind(estimates.roll(-i, -2)[1:])
            new_estimate = hd.bind(input, others)

            similarity = self.vsa.similarity(new_estimate, codebooks[i])
            if (activation == 'ABS'):
                similarity = torch.abs(similarity)
            elif (activation == 'NONNEG'):
                similarity[similarity < 0] = 0
            
            # Dot Product with the respective weights and sum
            # Update the estimate in place
            estimates[i] = hd.dot_similarity(similarity, codebooks[i].transpose(-2, -1)).sign()

        return estimates

    def resonator_stage_concur(self,
                               input: hd.VSATensor,
                               estimates: hd.VSATensor,
                               codebooks: hd.VSATensor or List[hd.VSATensor],
                               activation: Literal['NONE', 'ABS', 'NONNEG'] = 'NONE'):
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

        if (type(codebooks) == list):
            similarity = [None] * n
            output = [None] * n
            for i in range(n):
                similarity[i] = self.vsa.similarity(new_estimates[i], codebooks[i])
                if (activation == 'ABS'):
                    similarity[i] = torch.abs(similarity[i])
                elif (activation == 'NONNEG'):
                    similarity[i][similarity[i] < 0] = 0

                # Dot Product with the respective weights and sum
                output[i] = hd.dot_similarity(similarity[i], codebooks[i].transpose(-2,-1))

            output = torch.stack(output)
        else:
            similarity = self.vsa.similarity(new_estimates.unsqueeze(-2), codebooks)
            if (activation == 'ABS'):
                similarity = torch.abs(similarity)
            elif (activation == 'NONNEG'):
                similarity[similarity < 0] = 0

            # Dot Product with the respective weights and sum
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
# %%
