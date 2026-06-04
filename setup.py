from setuptools import setup, find_packages

setup(
    name="pyrocb",
    version="0.1.0",
    description="Feature generation and ML utilities for fire emissions modeling",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Jared Goldman",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xarray",
        "geopandas",
        "matplotlib",
        "rasterio",
        "scipy"
    ]
)