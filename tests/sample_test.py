import pytest
import sys
import importlib

def test_always_passes():
    """
    Standard 'True is True' tautology. 
    If this fails, your universe has collapsed (or pytest is broken).
    """
    assert True

def test_python_version():
    """
    Ensures the CI is actually running the version you specified (3.10).
    """
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 10

@pytest.mark.parametrize("package_name", [
    "numpy", 
    "xarray", 
    "rasterio", 
    "rioxarray",
    "fiona"
])
def test_dependencies_installed(package_name):
    """
    Dynamically checks if your core Conda dependencies are importable.
    """
    loader = importlib.util.find_spec(package_name)
    assert loader is not None, f"Dependency {package_name} is missing from the environment!"

def test_local_package_import():
    """
    Verifies 'pip install -e .' worked.
    """
    try:
        import pyrocb
    except ImportError:
        pytest.fail("Local package not found. Check if 'pip install -e .' ran in CI.")