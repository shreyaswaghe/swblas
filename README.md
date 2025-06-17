My "coding-gym" implementations of BLAS kernels, aiming for maximum performance on M1 Macbook Air. 

Features:
1. Standard CBLAS interface
2. Faster than ARMPL
3. Fully visible code & programming choices

Drawbacks:
1. Hypertuned to M1. Will likely not perform as well on other ARM processors.
2. Uses SSE2NEON as a SIMD translation layer. NEON intrinsics will likely be faster. This is a future goal.

Most Level 1 kernels attain 2.0x - 15.0x speedups over Arm Performance Libraries BLAS.
