Installation
============

`raccoon` is mainly implemented in python 3. It also uses some fortran subroutines that, in order to be called from python, are compiled with `numpy.f2py`.

Recommended: Create and activate an environment with e.g. conda and install `pip`:
```bash
conda create -n raccoon
conda activate raccoon
conda install pip
```

From source
-----------

The source code for `raccoon` can be downloaded from GitHub and installed by running
```bash
git clone https://github.com/mlafarga/raccoon.git
cd raccoon
python -m pip install .
```

To test the installation, the following command should display all the options for the script `raccoonccf`
```bash
# If installed from source, leave the source directory before testing this command
raccoonccf -h
```

To install in editable mode and/or include also the dependencies for the documentation and testing, run
```bash
python -m pip install -e .

# With docs dependencies
python -m pip install -e .[docs]

# With testing dependencies
python -m pip install -e .[test]

# With docs and testing dependencies
python -m pip install -e .[docs,test]
```

macOS note: zsh uses square brackets for globbing / pattern matching, so need to use e.g.
```bash
python -m pip install -e ."[docs,test]"
```