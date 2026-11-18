from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="warp_kernels",
    version="0.3.0",
    packages=["warp_kernels"],
    ext_modules=[
        CUDAExtension(
            name="warp_kernels._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/packed_gemv.cu",
                "csrc/packed_gemm.cu",
            ],
            depends=["csrc/bitpack.cuh", "csrc/common.cuh", "csrc/host_utils.h", "csrc/adaptive_prefetch_feedback.h"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
              
                "nvcc": ["-O3", "-std=c++17", "-lineinfo"],
            },
            extra_link_args=["-ldl"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
