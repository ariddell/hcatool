import pbr.version


__version__ = pbr.version.VersionInfo(
    'hcatool').version_string()

from .core import load, load_counts  # noqa
