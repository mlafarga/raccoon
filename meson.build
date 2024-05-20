project(
    'raccoon',
    'c',
    # ['c', 'fortran'],
    version: '0.0.1',
    # Cannot get dynamic version from SCM as with setuptools_scm?
    # Use a static version for now.
    # Some ideas (untested) below:
    # # ----------
    # version: run_command(
    #     'python3', # scm_version_snip
    #     [
    #         '-c',
    #         'from setuptools_scm import get_version;print(get_version(normalize=False))',
    #     ],
    #     check: true,
    # ).stdout().strip(),
    # # -----------
    # version: run_command(
    # # This should become `numpy/_version.py` in NumPy 2.0
    #   ['numpy/_build_utils/gitversion.py'],
    #   check: true).stdout().strip(),
    # # -----------
    # 
    # license: 'MIT',
    # 
    meson_version: '>= 1.1.0',
    # meson_version: '>=0.64.0',
    # 
    # default_options: [
        # 'cpp_std=c++11',
        # 'warning_level=2',
    # ],
)

add_languages('fortran')

# Python
py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

# -----------------------------------------------

# raccoon python modules
# ----------------------

# py_sources = [
#     'raccoon/__init__.py',
#     'raccoon/carmenesutils.py',
#     'raccoon/ccf.py',
#     'raccoon/espressoutils.py',
#     'raccoon/expresutils.py',
#     'raccoon/fitsutils.py',
#     'raccoon/harpsutils.py',
#     'raccoon/harpnsolarutils.py',
#     'raccoon/maroonxutils.py',
#     'raccoon/peakutils.py',
#     'raccoon/phoenixutils.py',
#     'raccoon/plotutils.py',
#     'raccoon/pyutils.py',
#     'raccoon/spectrographutils.py',
#     'raccoon/spectrumutils.py',
#     'raccoon/telluricutils.py',
#     'raccoon/scripts/ccfcompute.py',
#     'raccoon/scripts/maskcompute.py',
# ]

# py.install_sources(py_sources,
#   subdir: 'raccoon',
# )

# Seems like you don't need the above if have this line below
# Needed to get scripts raccoonccf and raccoonmask installed
install_subdir('raccoon', install_dir: py.get_install_dir() / 'raccoon', strip_directory: true)

# -----------------------------------------------

# raccoon fortran module (ccflibfort)
# -----------------------------------

# Numpy
incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

# f2py
incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

# CCF fortran module
ccf_source = custom_target('ccflibfortmodule.c',
  input : ['raccoon/ccflibfort.f'],
  output : ['ccflibfortmodule.c', 'ccflibfort-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'ccflibfort', '--lower']
)

inc_np = include_directories(incdir_numpy, incdir_f2py)

py.extension_module('ccflibfort',
  ['raccoon/ccflibfort.f', ccf_source],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : py_dep,
  subdir: 'raccoon',
  install : true,
)
