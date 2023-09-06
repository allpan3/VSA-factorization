# %%
import torch.nn as nn
from torch import Tensor
from typing import Literal, List
import torch
from vsa.vsa import VSA

# %%
class Resonator(nn.Module):

    def __init__(self, vsa:VSA, type="CONCURRENT", activation='NONE', iterations=100, argmax_abs=True, device="cpu"):
        super(Resonator, self).__init__()
        self.to(device)

        self.vsa = vsa
        self.device = device
        self.resonator_type = type
        self.iterations = iterations
        self.activation = activation
        self.argmax_abs = argmax_abs

    def forward(self, inputs, init_estimates):
        return self.resonator_network(inputs, init_estimates, self.vsa.codebooks, self.iterations, self.activation)

    def resonator_network(self, inputs: Tensor, init_estimates: Tensor, codebooks: Tensor or List[Tensor], iterations, activation):
        # Must clone, otherwise the original init_estiamtes will be modified
        estimates = init_estimates.clone()
        old_estimates = init_estimates.clone()
        for k in range(iterations):
            if (self.resonator_type == "SEQUENTIAL"):
                estimates = self.resonator_stage_seq(inputs, estimates, codebooks, activation)
            elif (self.resonator_type == "CONCURRENT"):
                estimates = self.resonator_stage_concur(inputs, estimates, codebooks, activation)
            elif (self.resonator_type == "COMBO"):
                if k == 0:
                    estimates = self.resonator_stage_concur(inputs, estimates, codebooks, activation)
                else:
                    estimates = self.resonator_stage_seq(inputs, estimates, codebooks, activation)
            if all((estimates == old_estimates).flatten().tolist()):
                break
            old_estimates = estimates.clone()

        # outcome: the indices of the codevectors in the codebooks
        outcome = self.vsa.cleanup(estimates, self.argmax_abs)

        return outcome, k 


    def resonator_stage_seq(self,
                            inputs: Tensor,
                            estimates: Tensor,
                            codebooks: Tensor or List[Tensor],
                            activation: Literal['NONE', 'ABS', 'NONNEG'] = 'NONE'):
        
        # Since we only target MAP, inverse of a vector itself

        for i in range(estimates.size(-2)):
            # Remove the currently processing factor itself
            rolled = estimates.roll(-i, -2)
            inv_estimates = torch.stack([rolled[j][1:] for j in range(estimates.size(0))])
            inv_others = self.vsa.multibind(inv_estimates)
            new_estimates = self.vsa.bind(inputs, inv_others)

            similarity = self.vsa.similarity(new_estimates, codebooks[i])
            if (activation == 'ABS'):
                similarity = torch.abs(similarity)
            elif (activation == 'NONNEG'):
                similarity[similarity < 0] = 0
            
            # Dot Product with the respective weights and sum
            # Update the estimate in place
            estimates[:,i] = self.vsa.multiset(codebooks[i], similarity, normalize=True)

        return estimates

    def resonator_stage_concur(self,
                               inputs: Tensor,
                               estimates: Tensor,
                               codebooks: Tensor or List[Tensor],
                               activation: Literal['NONE', 'ABS', 'NONNEG'] = 'NONE'):
        '''
        ARGS:
            inputs: `(*, d)`. d is dimension (b dim is optional)
            estimates: `(b, f, d)`. b is batch size, f is number of factors, d is dimension
        '''
        f = estimates.size(-2)
        if inputs.dim() == 1:
            b = 1
        else:
            b = inputs.size(0)
        d = inputs.size(-1)

        # Since we only target MAP, inverse of a vector itself

        # Roll over the number of estimates to align each row with the other symbols
        # Example: for factorizing x, y, z the stacked matrix has the following estimates:
        # [[z, y],
        #  [x, z],
        #  [y, x]]
        rolled = []
        for i in range(1, f):
            rolled.append(estimates.roll(i, -2))

        estimates = torch.stack(rolled, dim=-2)

        # First bind all the other estimates together: z * y, x * z, y * z
        inv_others = self.vsa.multibind(estimates)

        # Then unbind all other estimates from the input: s * (x * y), s * (x * z), s * (y * z)
        new_estimates = self.vsa.bind(inputs.unsqueeze(-2), inv_others)

        if (type(codebooks) == list):
            # f elements, each is VSATensor of (b, v)
            similarity = [None] * f 
            # Use int64 to ensure no overflow
            output = torch.empty((b, f, d), dtype=torch.int64, device=self.vsa.device)
            for i in range(f):
                # All batches, the i-th factor compared with the i-th codebook
                similarity[i] = self.vsa.similarity(new_estimates[:,i], codebooks[i]) 
                if (activation == 'ABS'):
                    similarity[i] = torch.abs(similarity[i])
                elif (activation == 'NONNEG'):
                    similarity[i][similarity[i] < 0] = 0

                # Dot Product with the respective weights and sum
                output[:,i] = self.vsa.multiset(codebooks[i], similarity[i], normalize=True)
        else:
            similarity = self.vsa.similarity(new_estimates.unsqueeze(-2), codebooks)
            if (activation == 'ABS'):
                similarity = torch.abs(similarity)
            elif (activation == 'NONNEG'):
                similarity[similarity < 0] = 0

            # Dot Product with the respective weights and sum
            output = self.vsa.multiset(codebooks, similarity, normalize=True).squeeze(-2)
        
        return output

    
# %%