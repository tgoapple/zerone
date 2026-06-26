"""ZEROne — M.I.P. Runtime."""
from setuptools import setup, find_packages

setup(
    name="zerone",
    version="0.2.0",
    description="Portable character runtime for LLM agents with persona enforcement.",
    author="Gary Clucas",
    packages=find_packages(exclude=["backup*", "data"]),
    python_requires=">=3.11",
    install_requires=[],
    extras_require={},
    entry_points={
        "console_scripts": ["zerone = cli:main"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
)
