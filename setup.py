from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="tree-discretization",
    version=0.1,
    # url="github.com/elianap",
    author="Eliana.pastor",
    author_email="eliana.pastor@polito.it",
    ext_modules=cythonize("criterion_tree_sklearn.pyx"),
    include_dirs=[numpy.get_include()],
)
