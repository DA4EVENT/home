import glob
import os
import os.path as osp

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))
_ext_sources = glob.glob(osp.join("src", "*.cpp")) + \
               glob.glob(osp.join("src", "*.cu"))
_ext_headers = glob.glob(osp.join("include", "*"))

requirements = ["torch>=1.4"]

exec(open("_version.py").read())

os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
setup(
    name="hats_cuda",
    version=__version__,
    author="Marco Cannici",
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="hats_cuda",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            },
            include_dirs=[osp.join(this_dir, "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
