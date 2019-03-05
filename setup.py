from setuptools import setup

setup(
    name="rivgraph",
    packages=['rivgraph'],
    version="0.1",
    description="Tools for topological analysis of fluvial networks from binary masks",
    author='Jon Schwenk',
    author_email='jonschwenk@gmail.com',
    url='https://github.com/jeueblah/rivgraph',
    keywords=['deltas', 'mask', 'topology', 'networks'],
#    install_requires=['pyinstrument_cext>=0.2.0'],
    include_package_data=True,
#    entry_points={'console_scripts': ['pyinstrument = pyinstrument.__main__:main']},
    zip_safe=False,
#    setup_requires=['pytest-runner'],
#    tests_require=['pytest'],
#    classifiers=[
#        'Development Status :: 4 - Beta',
#        'Environment :: Console',
#        'Environment :: Web Environment',
#        'Intended Audience :: Developers',
#        'License :: OSI Approved :: BSD License',
#        'Operating System :: MacOS',
#        'Operating System :: Microsoft :: Windows',
#        'Operating System :: POSIX',
#        'Programming Language :: Python :: 2.7',
#        'Programming Language :: Python :: 3.3',
#        'Topic :: Software Development :: Debuggers',
#        'Topic :: Software Development :: Testing',
#    ]
)
