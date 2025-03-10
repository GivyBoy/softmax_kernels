from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# ensure we have cuda arch list set
if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6"

nvcc_flags = [
    "-O3",  # Optimization level
    "--use_fast_math",  # Use fast math operations
    "--ptxas-options=-v",  # Verbose PTX assembly
    "--extended-lambda",
    "--expt-relaxed-constexpr",
    "-Xcompiler",
    "-fPIC",
    "-gencode=arch=compute_60,code=sm_60",  # Pascal
    "-gencode=arch=compute_70,code=sm_70",  # Volta
    "-gencode=arch=compute_75,code=sm_75",  # Turing
    "-gencode=arch=compute_80,code=sm_80",  # Ampere
    "-gencode=arch=compute_80,code=compute_80",  # PTX for future compatibility
]


setup(
    name="softmax_cuda",
    ext_modules=[
        CUDAExtension(
            name="softmax_cuda",
            sources=[
                "softmax_cuda.cpp",
                "softmax_cuda_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "--extended-lambda",
                    "--expt-relaxed-constexpr",
                    "--default-stream=per-thread",  # required for cooperative groups
                    "-Xcompiler",
                    "-fPIC",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
