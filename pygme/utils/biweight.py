#! /usr/bin/env python
#######################################################################
# Extracted and slightly adapted from ASTLIB and its astStats module
# http://astlib.sourceforge.net/docs/astLib/astLib.astStats-module.html
# From 2011 Matt Hilton
# Just replaced math by numpy
# 
# Used to estimate robust sigma and mean
#######################################################################
__version__ = '1.0.0 (December 25, 2012)'
import numpy as np
import math

__epsilon = 1.0e-20

def biweight_mean(dataList, tuningConstant=6.0, verbose=False): 
    """Calculates the biweight location estimator (like a robust average) of a list of 
    numbers. 
     
    @type dataList: list 
    @param dataList: input data, must be a one dimensional list 
    @type tuningConstant: float 
    @param tuningConstant: 6.0 is recommended. 
    @rtype: float 
    @return: biweight location 
     
    @note: Returns None if an error occurs.      
     
    """  
    C = tuningConstant 
    listMedian = np.median(dataList) 
    listMAD=MAD(dataList) 
    if listMAD!=0: 
        uValues=[] 
        for item in dataList: 
            uValues.append((item-listMedian)/(C*listMAD)) 
                 
        top=0           # numerator equation (5) Beers et al if you like 
        bottom=0        # denominator 
        for i in range(len(uValues)): 
            if np.abs(uValues[i]) <= 1.0: 
                top = top +((dataList[i]-listMedian) * (1.0-(uValues[i]*uValues[i])) * (1.0-(uValues[i]*uValues[i]))) 
                bottom = bottom+((1.0-(uValues[i]*uValues[i])) * (1.0-(uValues[i]*uValues[i]))) 
     
        CBI=listMedian+(top/bottom) 
         
    else: 
        if verbose :
            print """ERROR: biweight_mean() : MAD() returned 0.""" 
        return None 
     
    return CBI 
 
#--------------------------------------------------------------------------------------------------- 
def biweight_sigma(dataList, tuningConstant=9.0, verbose=False): 
    """Calculates the biweight scale estimator (like a robust standard deviation) of a list 
    of numbers.  
     
    @type dataList: list 
    @param dataList: input data, must be a one dimensional list 
    @type tuningConstant: float 
    @param tuningConstant: 9.0 is recommended. 
    @rtype: float 
    @return: biweight scale 
     
    @note: Returns None if an error occurs. 
         
    """  
    C = tuningConstant 
     
    # Calculate |x-M| values and u values 
    listMedian = np.median(dataList) 
    listMAD = MAD(dataList) 
    diffModuli=[] 
    for item in dataList: 
        diffModuli.append(np.abs(item-listMedian)) 
    uValues=[] 
    for item in dataList: 
        try: 
            uValues.append((item-listMedian)/(C*listMAD)) 
        except ZeroDivisionError: 
            if verbose :
                print """ERROR: biweight_sigma() : divide by zero error.""" 
            return None 
         
    top=0               # numerator equation (9) Beers et al 
    bottom=0 
    valCount=0  # Count values where u<1 only 
     
    for i in range(len(uValues)): 
        # Skip u values >1 
        if np.abs(uValues[i])<=1.0: 
            u2Term = 1.0-(uValues[i]*uValues[i]) 
            u4Term = np.power(u2Term, 4) 
            top = top + ((diffModuli[i]*diffModuli[i])*u4Term) 
            bottom = bottom + (u2Term*(1.0-(5.0*(uValues[i]*uValues[i])))) 
            valCount = valCount+1 
     
    top = np.sqrt(top) 
    bottom = np.abs(bottom) 
 
    SBI = np.power(float(valCount), 0.5)*(top/bottom) 
    return SBI

def MAD(dataList, verbose=False): 
    """Calculates the Median Absolute Deviation of a list of numbers. 
     
    @type dataList: list 
    @param dataList: input data, must be a one dimensional list 
    @rtype: float 
    @return: median absolute deviation 
     
    """ 
    listMedian = np.median(dataList) 
     
    # Calculate |x-M| values 
    diffModuli=[] 
    for item in dataList: 
        diffModuli.append(np.abs(item-listMedian)) 
    diffModuli.sort() 
   
    midValue=float(len(diffModuli)/2.0) 
    fractPart=np.modf(midValue)[0] 
   
    if fractPart==0.5:          # if odd number of items 
        midValue = np.ceil(midValue) 
         
    # Doesn't like it when handling a list with only one item in it!     
    if midValue<len(diffModuli)-1: 
        MAD = diffModuli[int(midValue)] 
         
        if fractPart!=0.5:      # if even 
            prevItem = diffModuli[int(midValue)-1] 
            MAD = (MAD+prevItem)/2.0 
   
    else: 
        MAD = diffModuli[0] 
         
    return MAD


##################################
## From LSL
##################################
def biweightMean(inputData):
	"""
	Calculate the mean of a data set using bisquare weighting.  
	
	Based on the biweight_mean routine from the AstroIDL User's Library.
	"""
	
	y = inputData.ravel()
	if type(y).__name__ == "MaskedArray":
		y = y.compressed()
	
	n = len(y)
	closeEnough = 0.03*np.sqrt(0.5/(n-1))
	
	diff = 1.0e30
	nIter = 0
	
	y0 = np.median(y)
	deviation = y - y0
	sigma = np.std(deviation)
	
	if sigma < __epsilon:
		diff = 0
	while diff > closeEnough:
		nIter = nIter + 1
		if nIter > __iterMax:
			break
		uu = ((y-y0)/(6.0*sigma))**2.0
		uu = np.where(uu > 1.0, 1.0, uu)
		weights = (1.0-uu)**2.0
		weights /= weights.sum()
		y0 = (weights*y).sum()
		deviation = y - y0
		prevSigma = sigma
		sigma = np.std(deviation, Zero=True)
		if sigma > __epsilon:
			diff = np.abs(prevSigma - sigma) / prevSigma
		else:
			diff = 0.0
			
	return y0

def robust_mean(inputData, Cut=3.0):
	"""
	Robust estimator of the mean of a data set.  Based on the 
	resistant_mean function from the AstroIDL User's Library.

	.. seealso::
		:func:`lsl.misc.mathutil.robustmean`
	"""

	data = inputData.ravel()
	if type(data).__name__ == "MaskedArray":
		data = data.compressed()

	data0 = np.median(data)
	maxAbsDev = np.median(np.abs(data-data0)) / 0.6745
	if maxAbsDev < __epsilon:
		maxAbsDev = (np.abs(data-data0)).mean() / 0.8000

	cutOff = Cut*maxAbsDev
	good = np.where( np.abs(data-data0) <= cutOff )
	good = good[0]
	dataMean = data[good].mean()
	dataSigma = math.sqrt( ((data[good]-dataMean)**2.0).sum() / len(good) )

	if Cut > 1.0:
		sigmaCut = Cut
	else:
		sigmaCut = 1.0
	if sigmaCut <= 4.5:
		dataSigma = dataSigma / (-0.15405 + 0.90723*sigmaCut - 0.23584*sigmaCut**2.0 + 0.020142*sigmaCut**3.0)

	cutOff = Cut*dataSigma
	good = np.where(  np.abs(data-data0) <= cutOff )
	good = good[0]
	dataMean = data[good].mean()
	if len(good) > 3:
		dataSigma = math.sqrt( ((data[good]-dataMean)**2.0).sum() / len(good) )

	if Cut > 1.0:
		sigmaCut = Cut
	else:
		sigmaCut = 1.0
	if sigmaCut <= 4.5:
		dataSigma = dataSigma / (-0.15405 + 0.90723*sigmaCut - 0.23584*sigmaCut**2.0 + 0.020142*sigmaCut**3.0)

	dataSigma = dataSigma / math.sqrt(len(good)-1)

	return dataMean


def robust_sigma(inputData, Zero=False):
	"""
	Robust estimator of the standard deviation of a data set.  
	
	Based on the robust_sigma function from the AstroIDL User's Library.
	"""

	data = inputData.ravel()
	if type(data).__name__ == "MaskedArray":
		data = data.compressed()

	if Zero:
		data0 = 0.0
	else:
		data0 = np.median(data)
	maxAbsDev = np.median(np.abs(data-data0)) / 0.6745
	if maxAbsDev < __epsilon:
		maxAbsDev = (np.abs(data-data0)).mean() / 0.8000
	if maxAbsDev < __epsilon:
		sigma = 0.0
		return sigma

	u = (data-data0) / 6.0 / maxAbsDev
	u2 = u**2.0
	good = np.where( u2 <= 1.0 )
	good = good[0]
	if len(good) < 3:
		print "WARNING:  Distribution is too strange to compute standard deviation"
		sigma = -1.0
		return sigma

	numerator = ((data[good]-data0)**2.0 * (1.0-u2[good])**2.0).sum()
	nElements = (data.ravel()).shape[0]
	denominator = ((1.0-u2[good])*(1.0-5.0*u2[good])).sum()
	sigma = nElements*numerator / (denominator*(denominator-1.0))
	if sigma > 0:
		sigma = math.sqrt(sigma)
	else:
		sigma = 0.0

	return sigma



