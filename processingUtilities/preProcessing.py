import pandas as pd
import numpy as np
import datetime
import scipy
import math


# unCompressor: where our assessments span more than 24 hours, multiple treatment options are possible
    # Inputs:
        # skipTreatment: Type of treatment applied to skipped days.
            # skipTreament = "None": no treatment applied, returns frame as is
            # skipTreament = "Dupe": skipped days are duped for the number of days skipped
            # skipTreament = "None": skipped days are duped for the number of days skipped, but dynamic duped values are replaced with Nan to be interpolated
        # dictOfTDFs: dictionary of dataframes for each entitiy
    # Returns:
        # dict of dataframes with treated skip days in 1 of three ways (skipTreament)
        
# NB: this preProcessing is only applicable in specific circumstances like the ones detailed in the paper.

def unCompressor(dictOfTDFs, skipTreatment = 'None'):
    
    if skipTreatment == "None":
        # final frame type 1
            # compressed skip days no interpolation
        return(dictOfTDFs)

    else:
        # final frame type 2 
            # duped skip days no interpolation
        dfDupeSkip = dict()
        for each in dictOfTDFs:
            tdf = dictOfTDFs[each].loc[dictOfTDFs[each].index.repeat(dictOfTDFs[each]['DaysInPeriod'])]
            tdf = tdf.reset_index(drop=True)
            dfDupeSkip[each] = tdf
            
        if skipTreatment == "Dupe":
            return(dfDupeSkip)
        
        
        elif skipTreatment == "NanDupe":
            # final frame type 3
                # duped skip days, duped values that are not static, are set to NAN
            nanCols = dfDupeSkip[111].columns[1:82]

            dfDupeNan = dict()

            for each in dfDupeSkip:


                tdf = dfDupeSkip[each]

                tseries = tdf.DaysInPeriod

                markList = [0]*len(tseries)

                for ind, val in tseries.iteritems():
                    if val > 1 and tseries[ind+val-1] != val:
                        markList[ind] = 1

                tdf['markColumn'] = markList

                tdf.loc[tdf.markColumn == 1, nanCols] = np.nan


                dfDupeNan[each] = tdf

            return(dfDupeNan)


# interpolator: interpolates dict of dataframes missing values in one of three ways
    # Inputs:
        # interpolationType: Type if interpolation applied to missing values
            # interpolateType = "None": None interpolation applied
            # interpolateType = "Linear": Linear interpolation applied
            # interpolateType = "Pad": Padding interpolation applied
        # dictOfTDFs: dictionary of dataframes for each entitiy
    # Returns:
        # dict of dataframes with interpolation performed as specifified.        

def interpolator(dictOfTDFs, interpolateType = 'None'):
    if interpolateType == "None":
        return(dictOfTDFs)


    elif interpolateType == "linear":
        rDict = dict()
        
        for each in dictOfTDFs:
            rDict[each] = dictOfTDFs[each].interpolate(method = 'linear', limit_direction = 'both')
            
        return(rDict)
    
    elif interpolateType == "pad":
        rDict = dict()

        for each in dictOfTDFs:
            rDict[each] = dictOfTDFs[each].interpolate(method = 'pad', limit_direction = 'forward')

        return(rDict)
    
    
#colTyper: takes dict of dataframes returns list of dynamic and catagorical columns and prints stats
    # Inputs:
        # dictOfTDFs: dictionary of dataframes for each entity
    # Returns:
        # names of collumns that change over time, and stay static, accross all entities, as two lists
    
    
def colTyper(dictOfTDFs):
    
    # dynamic cols
    dynamicCols = set()
    for each in dictOfTDFs:
        ts = dictOfTDFs[each].nunique(dropna=True).sort_values(ascending = False)
        tcols = ts[ts>1].index
        dynamicCols.update(tcols)
    dynamicCols = list(dynamicCols)


    # catagorical cols
    cataCols = list(set(dictOfTDFs[each].columns) - set(dynamicCols))

    print("dynamic covariables:", len(dynamicCols)-1)
    print("catagorical covariables:", len(cataCols))
    
    return(dynamicCols, cataCols)


# shortIDRemover: Removes all entities with less than a certain amount of entries
    # Inputs:
        # dictOfTDFs: dictionary of dataframes for each entity
        # shortStayLength: minimum length required per entity
    # Returns:
        # dict of dataframes with those shorter than shortStayLength removed
    
# NB: must be at shortest, 3 time stamps longer than than the longest input window length

def shortIDRemover(dictOfTDFs, shortStayLength):
    
    shortIds = list()
    for each in dictOfTDFs:
        if len(dictOfTDFs[each]) <shortStayLength:
            shortIds.append(each)

    print("removed",len(shortIds),'entities for being less than',shortStayLength,'periods long') 
    for key in shortIds:
        dictOfTDFs.pop(key)
    print("total remaining entities:",len(dictOfTDFs))
    
    return(dictOfTDFs)




# timeSeriesPercentTrainTestSplit: splits the full dict of dataframes into two containing the relevent data to be windowed by train and test windower.
    # Inputs:
        # dictOfTDFs: dictionary of dataframes for each entity
        # testPercent: percent of ending windows heldout for each entity
        # holdOutLength: Number of windows removed completly
        # MaxInputWindow: input window length
    # Returns:
        # Dict of dataframes per entity for train and test sets

# NB: this is the split process documented in the accompanying paper (Section 3.3) and the test dict contains the windowed days into the train set - the holdout length

def timeSeriesPercentTrainTestSplit(dictOfTDFs, testPercent, MaxInputWindow = 10, holdOutLength = 1):
    fdict = dict()
    ldict = dict()
    for iD in dictOfTDFs:
        dfLen = len(dictOfTDFs[iD])
        
        fr = range(0, dfLen - int(math.floor(testPercent*dfLen)))
        
        lr = range(dfLen - int(math.floor(testPercent*dfLen))-MaxInputWindow+holdOutLength,dfLen)
        
        fdict[iD] = dictOfTDFs[iD].iloc[fr]
        ldict[iD] = dictOfTDFs[iD].iloc[lr]
        
    return(fdict, ldict)


# standardizeOnTrain: standardize from the windows in the training set, apply to full dataframe.
    # Inputs:
        # dictOfTDFs: dictionary of dataframes for each entity
        # testPercent: percent of ending windows heldout for each entity
        # labelColumn: name of column containing the label binary values
        # holdOutLength: Number of windows removed completly
    # Returns:
        # standardized entity dict
        # dict of columns standard deviation across training data
        # dict of columns mean across training data

# NB: holdOutLength and testPercent must match the ones used in all following functions


def standardizeOnTrain(dictOfTDFs, testPercent, labelColumn, holdOutLenght = 1):
    
    strain, _ = timeSeriesPercentTrainTestSplit(dictOfTDFs, testPercent, MaxInputWindow = 1, holdOutLength = holdOutLenght)
    
    fullTrainDF = pd.concat([strain[each] for each in strain])

    stdDict = dict()
    meanDict = dict()

    for colName in fullTrainDF:
        stdDict[colName] = fullTrainDF[colName].std()
        meanDict[colName] = fullTrainDF[colName].mean()
        
    stdDict[labelColumn] = 1
    meanDict[labelColumn] = 0
        
    stdDictOfTDFs = dict()
    
    for each in dictOfTDFs:
        stdDictOfTDFs[each] = (dictOfTDFs[each] - meanDict)/stdDict
        
    return(stdDictOfTDFs, stdDict, meanDict)