import setuptools
from numpy.distutils.core import setup, Extension
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Fortran module
ccflibfort = Extension(name='raccoon.ccflibfort', sources=['raccoon/ccflibfort.f'])

setup(
# setuptools.setup(
    name='raccoon',
    version="0.0.1",
    description='Radial velocities and Activity indicators from Cross-COrrelatiON with masks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/mlafarga/raccoon',
    author="Marina Lafarga Magro",
    author_email="marina.lafarga@gmail.com",
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.7',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    keywords='CCF spectroscopy RV activity',
    package_dir={'raccoon': 'raccoon'},
    # packages=['raccoon'],
    packages=setuptools.find_packages(),  # better for scripts
    # python_requires='>=3.6',
    setup_requires=['numpy', 'scipy'],  # need for fortran
    install_requires=['numpy', 'scipy', 'astropy', 'pandas', 'lmfit', 'progress', 'matplotlib', 'tqdm', 'colorcet', 'cmocean', 'corner', 'emcee', 'h5py'],

    # package_data={
    #     # 'raccoon': ["data/*.dat", "data/*.mas", "data/*.fits", ],
    #     'raccoon': ['data/*.dat', "data/tellurics/*.dat", "data/mask/*.mas", "data/mask/*.pkl", "data/mask/*.dat", "data/phoenix/*.fits", ],
    # },
    include_package_data=True,  # get data from MANIFEST.in

    # scripts=['scripts/ccf_compute.py', 'scripts/mask_compute.py'],
    entry_points={
        'console_scripts': [
            'raccoonmask=raccoon.scripts.maskcompute:main',
            'raccoonccf=raccoon.scripts.ccfcompute:main',
            'raccoonlogL=raccoon.scripts.logLcompute:main',
            'raccoonlogLmcmc=raccoon.scripts.logLcomputemcmc:main',
        ],
    },

    ext_modules=[ccflibfort],
    )
