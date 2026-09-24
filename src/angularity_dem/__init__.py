"""Reproduction of Krengel, Chen & Kikumoto (2023), Comput. Geotech. 164, 105812.

"Effects of particle angularity on the bulk-characteristics of granular
assemblies under plane strain condition" - 2D polygonal DEM biaxial tests.
"""

__version__ = "0.1.0"


def main():
    from .cli import main as cli_main

    return cli_main()
