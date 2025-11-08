import torch
import torch.nn as nn
from mem_safe import mem_safe_forward

class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4), num_layers=2
        )
    def forward(self, x): return self.transformer(x)

def test_mem_safe_forward():
    model = SimpleTransformer()
    x = torch.randn(1, 16384, 64)  # 16K context
    
    output = mem_safe_forward(model, x, chunk_size=4096, use_checkpoint=False)
    assert output.shape == (1, 16384, 64)
    assert not torch.isnan(output).any()

def test_dynamic_chunk():
    model = SimpleTransformer()
    x = torch.randn(1, 32768, 64)
    output = mem_safe_forward(model, x, dynamic_chunk=True)
    assert output.shape[1] == 32768
