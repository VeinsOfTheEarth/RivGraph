from setuptools import setup

setup(
    name="rivgraph",
    packages=['rivgraph', 'rivgraph.deltas', 'rivgraph.rivers'],
    version="0.2",
    description="Tools for topological analysis of fluvial networks from binary masks",
    author='Jon Schwenk',
    author_email='jonschwenk@gmail.com',
    url='https://github.com/jonschwenk/rivgraph',
    keywords=['deltas', 'mask', 'topology', 'networks'],
    install_requires=['setuptools-git'],
    include_package_data=True,
    zip_safe=False
)
