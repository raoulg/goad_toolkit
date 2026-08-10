from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("goad-toolkit")
except PackageNotFoundError:  # source tree without an installed distribution
    __version__ = "unknown"
