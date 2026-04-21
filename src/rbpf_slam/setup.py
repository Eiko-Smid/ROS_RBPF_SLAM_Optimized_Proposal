#!/usr/bin/env python3

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import find_packages

setup_args = generate_distutils_setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)

setup(**setup_args)
