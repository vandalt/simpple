import simpple

def test_simpple_version():
    from importlib.metadata import version
    assert simpple.__version__ == version("simpple")
