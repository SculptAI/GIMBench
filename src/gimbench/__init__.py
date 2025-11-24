from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("gimbench")
except PackageNotFoundError:
    __version__ = "unknown"
