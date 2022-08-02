import torch.nn as nn
import torch
from .set_transformer import ISAB, PMA


class SetEncoder(nn.Module):
    def __init__(self, cfg, dim_input):
        super().__init__()
        self.bit16 = cfg.bit16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.bit16:
            self.Linear = nn.Linear(dim_input, 16*dim_input)

        self.selfatt = nn.ModuleList()
        self.selfatt1 = ISAB(16*dim_input, cfg.dim_hidden, cfg.num_heads, cfg.inducing_points, ln=cfg.norm_encoder)
        for _ in range(cfg.layers_encoder):
            self.selfatt.append(ISAB(cfg.dim_hidden, cfg.dim_hidden, cfg.num_heads, cfg.inducing_points, ln=cfg.norm_encoder))
        self.outatt = PMA(cfg.dim_hidden, cfg.num_heads, cfg.num_features, ln=cfg.norm_encoder)


    def float2bit(self, f, num_e_bits=5, num_m_bits=10, bias=127., dtype=torch.float32):
        ## SIGN BIT
        s = (torch.sign(f+0.001)*-1 + 1)*0.5 #Swap plus and minus => 0 is plus and 1 is minus
        s = s.unsqueeze(-1)
        f1 = torch.abs(f)
        ## EXPONENT BIT
        e_scientific = torch.floor(torch.log2(f1))
        e_scientific[e_scientific == float("-inf")] = -(2**(num_e_bits-1)-1)
        e_decimal = e_scientific + (2**(num_e_bits-1)-1)
        e = self.integer2bit(e_decimal, num_bits=num_e_bits)
        ## MANTISSA
        f2 = f1/2**(e_scientific)
        m2 = self.remainder2bit(f2 % 1, num_bits=bias)
        fin_m = m2[:,:,:,:num_m_bits] #[:,:,:,8:num_m_bits+8]
        return torch.cat([s, e, fin_m], dim=-1).type(dtype)

    def remainder2bit(self, remainder, num_bits=127):
        dtype = remainder.type()
        exponent_bits = torch.arange(num_bits, device = self.device).type(dtype)
        exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
        out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
        return torch.floor(2 * out)

    def integer2bit(self,integer, num_bits=8):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1, device = self.device).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) / 2 ** exponent_bits
        return (out - (out % 1)) % 2
    
    def forward(self, x):
        if self.bit16:
            x = self.float2bit(x)
            # Mirar shapes input
            x = x.view(x.shape[0], x.shape[1], -1)
            x = (x-0.5)*2    
        else:
            x = torch.relu(self.linearl(x))
        
        x = self.selfatt1(x)
        for layer in self.selfatt:
            x = layer(x)
        
        x = self.outatt(x)
        return x