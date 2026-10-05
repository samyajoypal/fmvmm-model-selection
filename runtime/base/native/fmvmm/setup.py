from setuptools import setup, find_packages

# Read the package description used by PyPI.
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

project_url = {
    "Github": "https://github.com/samyajoypal/fmvmm",
}

setup(
    name="fmvmm",
    version="3.0.2",
    author="Samyajoy Pal",
    author_email="palsamyajoy@gmail.com",
    license="MIT",
    description="Flexible Multivariate Mixture Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samyajoypal/fmvmm",
    project_urls=project_url,
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "conorm>=1.2,<2",
        "matplotlib>=3.7",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.6",
        "scipy>=1.9",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
