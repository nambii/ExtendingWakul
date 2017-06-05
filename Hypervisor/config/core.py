#!/usr/bin/python3
"""core
For converting  configure.yml to dict
"""

import os
import sys
import yaml
import argparse
from datetime import datetime
import dateutil.parser

def config(*subconfig):
    """Returns a dict that contains all of the settings.

    The settings are a combination of data in config.yaml and valid
    command line arguments.

    Args:
        *subconfig (str): Pass strings to return specific subsets of
            the config.

    Returns:
        dict: Software settings (config.yaml and command line args).

    """
    with open('config/configure.yaml', 'r') as stream:
        args = yaml.load(stream)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--node',
        '-n',
        help='The node ID.'
    )
    parser.add_argument(
        '--processes',
        '-p',
        help='The total number of processes.'
    )
    # Store command line arguments in a dict
    cl_args = parser.parse_args()
    cl_args_dict = vars(cl_args)
    # Combine
    args.update(cl_args_dict)
    # Find subconfig if argument is passed
    for s in subconfig:
        try:
            args = args[s]
        except:
            pass
    # Return
    return args
