import pathlib

import pkg_resources
from setuptools import setup, find_packages

PKG_NAME = "transic"
VERSION = "0.0.1"
EXTRAS = {}


def _read_file(fname):
    with pathlib.Path(fname).open() as fp:
        return fp.read()


def _read_install_requires():
    with pathlib.Path("requirements.txt").open() as fp:
        return [
            str(requirement) for requirement in pkg_resources.parse_requirements(fp)
        ]


setup(
    name=PKG_NAME,
    version=VERSION,
    author="TRANSIC Developers",
    description="research project",
    long_description=_read_file("README.md"),
    long_description_content_type="text/markdown",
    keywords=["Robotics", "Reinforcement Learning", "Machine Learning"],
    license="Apache License, Version 2.0",
    packages=find_packages(include=f"{PKG_NAME}.*"),
    include_package_data=True,
    zip_safe=False,
    entry_points={"console_scripts": []},
    install_requires=_read_install_requires(),
    python_requires="==3.8.*",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Robotics",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
    ],
)
