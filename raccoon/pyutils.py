#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np

###############################################################################


# Files and directories

def get_basename(s):
    """Remove path and extension of string `s` to get the "basename".

    ```
    >>> os.path.basename('path/to/myfile.txt')
    'myfile.txt'
    >>> os.path.splitext('path/to/myfile.txt')
    ['path/to/myfile', '.txt']
    >>> get_basename('path/to/myfile.txt')
    'myfile'
    ```
    """
    return os.path.basename(os.path.splitext(s)[0])


def make_dir(d):
    """Create directory `d` if it does not exist."""
    if not os.path.exists(d): os.makedirs(d)
    return

# -----------------------------------------------------------------------------


# Variable types

def isfloat(x):
    """Return True if can convert `x` to float, return False otherwise.
    https://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python
    """
    try:
        float(x)
        return True
    except ValueError:
        return False


def isfloatnum(x):
    """Check if `x` is a "valid" float, i.e. a real number, not a nan or inf. If it is a boolean, it is assumed that it is a float, either 0 or 1."""
    if isfloat(x):
        # Check for nan and inf
        if np.isfinite(float(x)):
            return True
        else:
            return False
    else:
        return False

# -----------------------------------------------------------------------------


# System

def get_distro():
    """Get OS name."""
    with open('/etc/os-release', 'r') as f:
        d = f.read().splitlines()
    distro = [l for l in d if l.startswith('ID=')][0].replace('ID=', '').lower()
    return distro

# -----------------------------------------------------------------------------


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Custom formatter class for `argparse` to be able to show default values of options and keep formatting of `description` and `epilog` text"""
    pass

# -----------------------------------------------------------------------------


# Commands

def save_command_current(dirout, command, filout='cmdlast.txt'):
    """
    Save current command.
    Usually:
        dirout = args.dirout
        command = sys.argv
    """
    with open(os.path.join(dirout, filout), 'w') as fout:
        fout.write(' '.join(command))
    return


def save_command_history(dirout, command, filout='cmdhistory.txt'):
    """
    Save current command in commands history.
    Usually:
        dirout = args.dirout
        command = sys.argv
    """
    with open(os.path.join(dirout, filout), 'a') as fout:
        fout.write(' '.join(command))
        fout.write('\n')
    return


def save_commandlineargs(dirout, args, filout='cmdlastargs.txt'):
    """
    Save current command line arguments.
    Usually:
        dirout = args.dirout
        args = args
    """
    with open(os.path.join(dirout, filout), 'w') as fout:
        # for k,v in sorted(args.__dict__.iteritems()):
        for k, v in sorted(args.__dict__.items()):
            fout.write('{}: {}\n'.format(k, v))


def save_command_current_hist_args(dirout, command, args):
    """
    Uses `save_command_current`, `save_command_history`, `save_commandlineargs`.
    Usually:
        dirout = args.dirout
        command = sys.argv
        args = args
    """
    save_command_current(dirout, command, filout='cmdlast.txt')
    save_command_history(dirout, command, filout='cmdhistory.txt')
    save_commandlineargs(dirout, args, filout='cmdlastargs.txt')
    return
