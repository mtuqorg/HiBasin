import setuptools

setuptools.setup(
    name="hibasin",
    version="1.0",
    author="Jinyin Hu et al.",
    author_email="jinyin.hu@anu.edu.au",
    description="Hiarchy Bayesian Source Inversion",
    long_description="Hierarchical Bayesian Source Inversion for seismic moment tensor (MT), single force (SF), or joint MT and SF with incoporating uncertainty estimates for data noise and theory error",
    long_description_content_type="text/markdown",
    url="https://github.com/mtuqorg/HiBasin",
    project_urls={
        "Bug Tracker": "https://github.com/mtuqorg/HiBasin/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords=[
        "seismology"
    ],
    python_requires='>=3.0',
    install_requires=[
        "emcee",
        "numpy<2",
        "corner",
        "pyrocko",
    ],
    packages=setuptools.find_packages(),
)