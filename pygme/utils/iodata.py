######################################################################
# Adapted from Erik Tollerud package astropysics, in utils/io,py
# Directly gets the data file from the sub-module 
# The data file should be under the directory ./data
######################################################################
def get_package_data(dataname, asfile=False):
    """
    Use this function to load data files distributed with the pygme
    source code.

    :param str dataname:
        The name of a file in the package data directory.
    :returns: The content of the requested file as a string.

    """
    import inspect
    import os
    dirpath = os.path.dirname(inspect.stack()[1][1])
    datapath = dirpath+'/data/'+dataname
    print(datapath)
    return get_data(datapath, asfile=asfile)

def get_data(datapath, asfile=False):
    """
    Retrieves a data file from a local file

    :param str datapath:
        The path of the data to be retrieved. 
    :param bool asfile:
        If True, a file-like object is returned that can be used to access the
        data. Otherwise, a string with the full content of the file is returned.
    :param localfn:
        The filename to use for saving (or loading, if the file is present)
        locally. If it is None, the filename will be inferred from the URL
        if possible. This file name is always relative to the astropysics data
        directory (see :func:`astropysics.config.get_data_dir`). This has no
        effect if :func:`set_data_store` is set to False.

    :returns: A file-like object or a loaded numpy array (with loadtxt)

    :raises IOError:
        If the datapath is requested as a local data file and not found.

    """
    import os

    ## The file is a local file - try to get it
    if not os.path.isfile(datapath) :
        print("The file %s you are trying to access does not exist" %(datapath))
        raise IOError
    fn = datapath
    if asfile:
        return open(fn)
    else:
        import numpy as np
        return np.loadtxt(fn)
