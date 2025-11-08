<div align="center">

# **mem-safe**  
### **128K Context on 1 GPU**

> **No OOM. Full Training. Auto Chunk.**

![PyPI](https://img.shields.io/pypi/v/mem-safe?color=orange)
![Python](https://img.shields.io/badge/python-≥3.8-blue)

</div>

---

**Author**: **Hari Tedjamantri**  
**Email**: haryganteng06@gmail.com  
**X**: [@haritedjamantri](https://x.com/haritedjamantri)

---

## **Install**

```bash
pip install mem-safe
```
## Quick Start

```bash
from mem_safe import mem_safe_forward

# Training 128K
output = mem_safe_forward(model, x, use_checkpoint=True)

# Inference
output = mem_safe_forward(model, x, use_checkpoint=False)
```

## Results (A100 40GB)


Context | Normal | MEM-SAFE
--------|--------|----------
32K     | OK     | OK
64K     | OOM    | OK
128K    | OOM    | OK
Memory  | 38GB   | 18GB

## Citation

@software{mem-safe-2025,

  author = {Hari Tedjamantri},
  
  title = {mem-safe: Memory-Efficient Long-Context Processing},
  
  year = {2025},
  
  url = {https://github.com/ebc-clip/mem-safe}
}
<div align="center">
Made with  by **Hari Tedjamantri**
</div>
```


