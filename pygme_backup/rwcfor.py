#! /usr/bin/python
# -*- coding: iso-8859-15 -*-
"""
This module allows to read and write fortran written binaries
"""

"""
Importing the most important modules
This module requires :
 - numpy version>=1.0
"""
import numpy as np
from numpy import float64 as floatMGE
from numpy import float64 as floatG
from numpy import float32 as floatsG
from numpy import int32 as intG
from numpy import int16 as intsG
from numpy import uint as uintG

__version__ = '1.0.8 (10 January, 2012)'
# Version 1.0.8 : - Cleaning for release pygme
# Version 1.0.6 : - Got print_msg from snapshot
# Version 1.0.5 : - Added some checks and verbose issues
# Version 1.0.4 : - Changed the struct.calcsize to itemsize to follow numpy specifics
# Version 1.0.3 : - Added specific MGE float
# Version 1.0.2 : - Removed specifics in the "except"
# Version 1.0.1 : - Added end of file function
# Version 1.0.0 : - First version adapted from the version 2.1.4 of pmsphsf.py
################################################################
#  Reading and Writing Fortran binary files
################################################################

#############################
## Data formats
#############################
sizeintsG = (np.zeros(1,dtype=np.int16)).itemsize   ## corresponds to intsG
sizeintG = (np.zeros(1,dtype=np.int32)).itemsize   ## corresponds to intG
sizefloatsG = (np.zeros(1,dtype=np.float32)).itemsize   ## corresponds to floatsG
sizefloatG = (np.zeros(1,dtype=np.float64)).itemsize   ## corresponds to floatG
## sizeint = struct.calcsize('i')
## sizelong = struct.calcsize('l')
## sizefloat = struct.calcsize('f')
## sizedouble = struct.calcsize('d')

#############################
## End of file
#############################
def end_of_file(file=None) :
    """ Checking if we are at the end of the file
        Return True if it is the end
    """
    p1 = file.tell()
    file.seek(0,2)
    p2 = file.tell()
    file.seek(p1)

    if p1 == p2:
        status = True
    else:
        status = False

    return status

#############################
## read_for_fast
#############################
def read_for_fast(file=None, numbers=[1], type=floatsG, arch=0, verbose=0) :
    """
    Reading data from Fortran binary files

    @param file: name of file for reading data
    @type  file: string (default is None)
    @param numbers: numbers of data of a certain type to be read
    @type  numbers: list (default is [1])
    @param type: type of data
    @type  type: list of num types (default is float)
    @param arch: byte order for reading data (0: little-endian, std. size & alignment ; 1: big-endian, std. size & alignment)
    @type  arch: integer (0 or 1 ; default is 0)

    @return: status, data
    @rtype: integer, list
    """

    data = []
##    fakedata = [0]
##    for i in range(len(numbers)) :
##       tempdata = np.zeros(numbers[i], type[i])
##       fakedata.append(tempdata)
##    fakedata.append(0)

    ## Reading the first item: the size
    try :
        data.append(np.fromfile(file,dtype=intG,count=1)[0])
    except  :
        return -1, [0,[0],0]   # return the status and add the 2nd+3rd to data

    ## Reading the second item: the data
    dataprop = zip(type,numbers)

    sizetot = 0
    for i in dataprop:
        #        try :
        if (i!=0):
            if(i[0]=='string'):
                data.append(file.read(i[1]))
                sizetot += i[1]
            else:
                tmp = np.fromfile(file,*i)
                sizetot += i[1] * (np.zeros(1,dtype=i[0])).itemsize
                if (len(tmp) == i[1]):
                    if (arch == 1):
                        tmp = tmp.byteswap() #byteswapping if necessary
                    if (len(tmp) == 1):
                        data.append(tmp[0])
                    else:
                        data.append(tmp)
                else:
                    #        except EOFError :
                    return -2, data+[0]   # return the status and add the 3rd to data

    if verbose :
        print "Read-FAST: Reading %s bytes" %(sizetot)

    ## Reading the third item: the size again
    try :
        data.append(np.fromfile(file,dtype=intG,count=1)[0])
    except :
        data.append(0)
        return -3, data

    if(data[-1]!=data[0]):
        print data[1]
        print 'data[-1]!=data[0]!'
        print 'data[0]',data[0]
        print 'data[-1]',data[-1]
        print 'problem while reading binary file!'
        return -4,data
    return 0, data   # return the status (0=ok) and data

#############################
## write_for_fast
#############################
def write_for_fast(file, data, size=[4], arch=0, verbose=0) :
    """
    Writing data to Fortran binary files

    @param file: name of file for writing data
    @type  file: string
    @param data: data to write
    @type  data: list
    @param size: size of data to write in bytes
    @type  size: list (default is [4])
    @param arch: byte order for reading data (0: little-endian, std. size & alignment ; 1: big-endian, std. size & alignment)
    @type  arch: integer (0 or 1 ; default is 0)

    """
#    if len(type) != len(size) :
#        print "ERROR: Lengths of 'type' and 'size' not identical: cannot read the data"
#        return 1

    sizetot = np.array([sum(size)],dtype=intG)
    if verbose :
        print "Write-FAST: Writing %s bytes" %(sizetot)
    sizetot.tofile(file)

    if (sizetot != 0):
        for da in data:
            da.tofile(file)

    sizetot.tofile(file)

    return 0

################################################################
#  Reading and Writing C binary files
################################################################

#############################
## read_C_fast
#############################
def read_C_fast(file=None, numbers=[1], type=floatsG, arch=0) :
    """
    Reading data from Fortran binary files

    @param file: name of file for reading data
    @type  file: string (default is None)
    @param numbers: numbers of data of a certain type to be read
    @type  numbers: list (default is [1])
    @param type: type of data
    @type  type: list of num types (default is float)
    @param arch: byte order for reading data (0: little-endian, std. size & alignment ; 1: big-endian, std. size & alignment)
    @type  arch: integer (0 or 1 ; default is 0)

    @return: status, data
    @rtype: integer, list
    """

    data = []
##    fakedata = [0]
##    for i in range(len(numbers)) :
##       tempdata = np.zeros(numbers[i], type[i])
##       fakedata.append(tempdata)
##    fakedata.append(0)

    ## Reading the second item: the data
    dataprop = zip(type,numbers)

    for i in dataprop:
        #        try :
        if (i!=0):
            if(i[0]=='string'):
                data.append(file.read(i[1]))
            else:
                tmp = np.fromfile(file,*i)
                if (len(tmp) == i[1]):
                    if (arch == 1):
                        tmp = tmp.byteswap() #byteswapping if necessary
                    if (len(tmp) == 1):
                        data.append(tmp[0])
                    else:
                        data.append(tmp)
                else:
                    #        except EOFError :
                    return -2, data+[0]   # return the status and add the 3rd to data

    return 0, data   # return the status (0=ok) and data

#############################
## write_C_fast
#############################
def write_C_fast(file, data, size=[4], arch=0) :
    """
    Writing data to Fortran binary files

    @param file: name of file for writing data
    @type  file: string
    @param data: data to write
    @type  data: list
    @param size: size of data to write in bytes
    @type  size: list (default is [4])
    @param arch: byte order for reading data (0: little-endian, std. size & alignment ; 1: big-endian, std. size & alignment)
    @type  arch: integer (0 or 1 ; default is 0)

    """
    sizetot = np.array([sum(size)])

    if (sizetot != 0):
        for i in data:
            i.tofile(file)

    return 0
