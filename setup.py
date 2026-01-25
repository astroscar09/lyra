from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lyra",
    version="0.1.0",
    author="Oscar A. Chavez Ortiz",
    description="A Python package for Lyman-alpha Galaxy Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "pyyaml",
        "torch",
        "sbi",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
