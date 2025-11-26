import importlib.metadata

try:
  __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
  # When the project is imported from source (e.g., Modal mounting the repo)
  # the package metadata might not be installed. Fall back to a placeholder.
  __version__ = "0.0.0"