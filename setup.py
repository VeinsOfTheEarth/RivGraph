from setuptools import setup, find_packages
from rivgraph import __version__

setup(
    name="rivgraph",
    packages=find_packages(),
    version=__version__,
    description="Tools for topological analysis of fluvial networks from binary masks",
    author='Jon Schwenk',
    author_email='jonschwenk@gmail.com',
    url='https://github.com/jonschwenk/rivgraph',
    keywords=['deltas', 'mask', 'topology', 'networks'],
    classifiers = ['Programming Language :: Python :: 3.6'],
    include_package_data=True,
    zip_safe=False
)
