import simpple


def test_simpple_version():
    from importlib.metadata import version

    # For editable installs, you might just need to re-install if this fails
    assert simpple.__version__ == version("simpple")
