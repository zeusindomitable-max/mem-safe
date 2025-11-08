import torch
import torch.utils.checkpoint as checkpoint
from typing import Optional
import warnings

def estimate_safe_chunk_size(
    model,
    sample_input: torch.Tensor,
    max_chunk: int = 8192,
    safety_factor: float = 0.8
) -> int:
    """
    Estimate safe chunk size based on GPU memory.
    """
    try:
        torch.cuda.empty_cache()
        test_chunk = sample_input[:, :512, :].contiguous()
        with torch.no_grad():
            _ = model.transformer(test_chunk)
        peak_mem = torch.cuda.max_memory_allocated()
        total_mem = torch.cuda.get_device_properties(0).total_memory
        available = total_mem - peak_mem
        safe_chunk = int((available * safety_factor / peak_mem) * 512)
        return min(safe_chunk, max_chunk)
    except Exception as e:
        warnings.warn(f"Memory estimation failed: {e}. Using default 2048.")
        return 2048

def mem_safe_forward(
    model,
    x: torch.Tensor,
    chunk_size: Optional[int] = None,
    use_checkpoint: bool = True,
    dynamic_chunk: bool = True,
    verbose: bool = False
):
    """
    Memory-Safe Forward: Handle 128K+ context on 1 GPU.
    
    Args:
        model: nn.Module with .transformer
        x: Input tensor (B, L, D)
        chunk_size: Fixed chunk size
        use_checkpoint: Enable gradient checkpointing (training)
        dynamic_chunk: Auto-estimate chunk size
        verbose: Print memory stats
    """
    if dynamic_chunk and chunk_size is None:
        chunk_size = estimate_safe_chunk_size(model, x)
    elif chunk_size is None:
        chunk_size = 4096

    chunks = x.split(chunk_size, dim=1)
    outputs = []

    for i, chunk in enumerate(chunks):
        def forward_chunk(chunk=chunk):
            return model.transformer(chunk)

        if use_checkpoint and x.requires_grad:
            hidden = checkpoint.checkpoint(forward_chunk)
        else:
            with torch.no_grad():
                hidden = forward_chunk()

        outputs.append(hidden)

        if verbose:
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"[Mem-Safe] Chunk {i+1}/{len(chunks)} | Size: {chunk.shape[1]} | Mem: {mem_gb:.2f} GB")

    return torch.cat(outputs, dim=1)
