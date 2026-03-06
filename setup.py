from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bench-stat",
    version="1.0.0",
    author="Eastig",
    description="Statistical Benchmarking Analysis Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eastig/bench_stat",
    py_modules=["bench_stat"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Testing",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=5.0",
            "pytest-cov>=2.8",
        ],
    },
)
