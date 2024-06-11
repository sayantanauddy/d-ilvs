import os
import datetime
import json
import logging
import inspect
import numpy as np
from pprint import pprint
import itertools
import glob
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    """
    Encodes numpy arrays for writing to a file
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class Dictobject(object):
    """
    Used for accessing dict items as attributes
    """
    def __init__(self, d):
        self.__dict__ = d

def transform(multilevelDict):
    '''
    Converts keys from '1' to 1 (str->int) for a nested dict
    '''
    return {int(key) if isinstance(key,str) and key.isdigit() else key: (transform(value) if isinstance(value, dict) else value) for key, value in multilevelDict.items()}

def list2np(in_dict):
    
    for k, v in in_dict.items():
        if isinstance(v, list):
            in_dict[k] = np.array(v)
        elif isinstance(v, dict):
            list2np(v)
        else:
            in_dict[k] = v
    return in_dict

def read_dict(json_path):
    """Reads a json file into a dict and returns it.
    Args:
        json_path (str): Path
    Returns:
        dict: Python dict
    """
    json_dict = None
    with open(json_path, "r") as json_file:
        json_dict = json.load(json_file)
    return json_dict

def write_dict(json_path, json_dict):
    """Writes dict to json file.
    Args:
        json_path (str): Path
        json_dict (dict): Python dict
    """
    with open(json_path, "w") as json_file:
        json.dump(json_dict, json_file, indent=4)

def read_np_dict(json_path):
    """Reads a json file into a dict and returns it.
       If values in the nested dict are lists, they
       are converted into numpy arrays.
       If keys are ints in string format, they 
       are converted to ints.
    Args:
        json_path (str): Path
    Returns:
        dict: Python dict
    """
    json_dict = read_dict(json_path)

    # Fix str keys which are actually numbers
    json_dict = transform(json_dict)

    # Convert list to np array in values
    json_dict = list2np(json_dict)

    return json_dict

def write_np_dict(json_path, json_dict):
    """Writes dict containing numpy arrays to json file.
    Args:
        json_path (str): Path
        json_dict (dict): Python dict
    """
    with open(json_path, "w") as json_file:
        json.dump(json_dict, json_file, indent=4, cls=NumpyArrayEncoder)

def read_numpy(numpy_file_path):
    """Reads a numpy archive file and returns the array
    Args:
        numpy_file_path (str): Path of the numpy archive file
    Returns:
        numpy arr: The array read from the file.
    """
    arr = np.load(numpy_file_path, mmap_mode='r')
    return arr

def get_id():
    """Creates a unique string ID based on the name of the calling
       script and the timestamp.
    Returns:
        str: Identifier string
    """
    # Get the name of the script from which get_id is called
    # https://www.stefaanlippens.net/python_inspect/
    caller_filename = os.path.basename(inspect.stack()[1][1])[0:-3]

    # Timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create an ID
    identifier = f"{caller_filename}_{timestamp}"

    return identifier

def custom_logging_setup_simple(args):
    """
    Sets up logging directories, saves commandline args to a file,
    and enables saving a dump of the console logs
    """

    # Set-up output directories
    identifier = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    save_folder = os.path.join(args.log_dir, args.description, identifier)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)

    return save_folder, identifier

def custom_logging_setup(args):
    """
    Sets up logging directories, saves commandline args to a file,
    and enables saving a dump of the console logs
    """

    # Set-up output directories
    identifier = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    identifier = f"{identifier}_seed{args.seed}"
    save_folder = os.path.join(args.log_dir, args.description, identifier)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Make a folder for storing trained models
    model_dir = os.path.join(save_folder, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)

    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    return save_folder, identifier

def count_lines(file_name):
    """
    Counts number of non-empty lines in the file `file_name`
    Args:
        file_name (str): Path to file
    Returns:
        int: Number of non-empty lines
        list(str): List of strings, one str for each line
    """
    with open(file_name) as f:
        lines = [line.rstrip() for line in f.readlines()]
    nonblanklines = [line for line in lines if line]
    return len(nonblanklines), nonblanklines