import pandas as pd
import numpy as np
import datetime
import scipy
import random
import math






# trainWindower: returns the rolling training windows as described in section 3.3 and crossvalidation fold data, using coulumn names to represent sequential data for a single dataframe
    # Inputs:
        # dataFrame: dataframe for single entity ordered by time
        # windowLength: input window length
        # dynamicCols: dynamic column names list as created by colTyper
        # cataCols: static column names list as created by colTyper
        # labelCol: name of column containing the label binary values
    # Returns:
        # dataframe of training windows for a single entitity, with columns named: name_lag_0, name_lag_1 ....  with the number representing how many iterations back in time that variable represents 
        # np.array of the labels corresponding to each window in the dataframe
        
# NB: this function needs to be optimised, it currently is extermely slow.
# NB: This is the dataframe/flat version as used by all non iterative models such as ours.


def windower(dataFrame, windowLength, dynamicCols, cataCols, labelCol):

    lagCols = list()
    for colName in dynamicCols:
        lagCols += [colName + "_lag_" + str(no) for no in range (windowLength)]

    allCols = lagCols + cataCols


    rows = list()
    trainLabels = list()

    tempDF = dataFrame.reset_index( drop = True)
    noWindows = len(tempDF) - windowLength
    j = 0
    if noWindows >= 1:

        for i in range(0,noWindows):

            startIDX = j
            endIDX = j+windowLength


            row = list()
            for colName in dynamicCols:
                row += list(tempDF[colName][startIDX:endIDX])[::-1]

            predVal = tempDF[labelCol][endIDX]

                
            if np.isnan(predVal) == False:

                rows.append(row+ list(tempDF[cataCols].iloc[0]))
                trainLabels.append(predVal)

            j = j+1


    trainDF = pd.DataFrame(rows, columns = allCols) 

    return(trainDF,trainLabels)






# arrayWindower: returns a single dataframes rolling windows
    # Inputs:
        # dataFrame: dataframe for single entity ordered by time
        # windowLength: input window length
        # predictionVariable: name of column containing the label binary values
        # timeFirst: indication for if window arrays should be time or feature first
            # timeFirst = True: output.shape = (window_number, timestep, feature)
            # timeFirst = False: output.shape = (window_number, feature, timestep)
            # True is used by RNN-BOF
        # sliding: indication for if windows should be sliding or non-overlapping
            # True is used by RNN-BOF
        # Univariate: Use True if data is univariate
        # removeNaNY: If true, removes windows and labels if the label is missing
            # redundant due to other pre-processing steps
    # Returns:
        # 3D np.array representing windows, timesteps and features
        # np.array of the labels corresponding to each window in the dataframe

def arrayWindower(dataFrame, windowLength, targetVariable, timeFirst, sliding, univariate, removeNaNY):
    rows = list()
    trainLabels = list()
    tempDF = dataFrame.reset_index(drop = True)
    
    if sliding == True:
        noWindows = len(tempDF) - windowLength
    else:
        noWindows = math.floor(len(tempDF)/(windowLength+1))-1


    j = 0
    
    if noWindows >=1:
        for i in range(0, noWindows):
            startIDX = j
            endIDX = j+windowLength
            
            row = list()
            
            
            if univariate == False:
                for colName in tempDF:
                    row.append(np.array(tempDF[colName][startIDX:endIDX]))
            else:
                    row.append(np.array(tempDF[targetVariable][startIDX:endIDX]))

                
            
            predVal = tempDF[targetVariable][endIDX]
                
            if removeNaNY == True:
                if np.isnan(predVal) == False:
                    if timeFirst == True:
                        rows.append(np.asarray(list(map(np.array, zip(*row)))))
                    else:
                        rows.append(np.asarray(row))

                    trainLabels.append(tempDF[targetVariable][endIDX])
                    
            else: 
            
                if timeFirst == True:
                    rows.append(np.asarray(list(map(np.array, zip(*row)))))
                else:
                    rows.append(np.asarray(row))

                trainLabels.append(tempDF[targetVariable][endIDX])

            
            
            
            if sliding == True:
                j = j+1
            else:
                j = j +windowLength+1

            
    
    return(np.asarray(rows), (np.asarray(trainLabels)))





# arrayTrainWindowerRolling: returns the rolling trainign windows as described in section 3.3 and crossvalidation fold data, using arrays to represent sequential windows
    # Inputs:
        # trainingDfDict: training data from timeSeriesPercentTrainTestSplit as a dict of dataframes
        # numberOfFolds: number of folds for sequential cross validation (section 3.3 and 4.4). will create numberOfFolds + 1 pools for numberOfFolds sequential evaluations
        # windowLength: input window length
        # predictionVariable: name of column containing the label binary values
        # timeFirst: indication for if window arrays should be time or feature first
            # timeFirst = True: output.shape = (window_number, timestep, feature)
            # timeFirst = False: output.shape = (window_number, feature, timestep)
            # True is used by RNN-BOF
        # sliding: indication for if windows should be sliding or non-overlapping
            # True is used by RNN-BOF
        # Univariate: Use True if data is univariate
        # removeNaNY: If true, removes windows and labels if the label is missing
            # redundant due to other pre-processing steps
    # Returns:
        # 3D np.array representing windows, timesteps and features
        # np.array of the labels corresponding to each window in the dataframes
        # array of arrays to be used by the custom cross validation function, containg the DF index for each cross validations folds

        
def arrayTrainWindowerRolling(trainingDfDict, numberOfFolds, windowLength, predictionVariable, timeFirst, sliding, univariate, removeNaNY):
    
    arrayList = list()
    labelList = list()
    foldList =list()
    
    for each in trainingDfDict:
        
        ta, tl = arrayWindower(trainingDfDict[each], windowLength, predictionVariable, timeFirst, sliding, univariate, removeNaNY)
        
        b = np.array_split(np.arange(0,len(ta)), numberOfFolds)
        
        foldNumberList = list()
        for i in range(len(b)):
            for each in b[i]:
                foldNumberList.append(i)
                
                
        arrayList.append(ta)
        labelList.append(tl)
        foldList.append(foldNumberList)
        
    
    
    
    x_train = np.concatenate(arrayList, axis = 0)
    y_train = np.concatenate(labelList, axis = 0)
    cv_group = np.concatenate(foldList, axis = 0)
    
    foldDf = pd.DataFrame(cv_group, columns = ['foldGroup'])
    
    trainRows = list()
    testRows = list()
    
    for i in range(0, numberOfFolds - 1):
        trainRows.append(list(foldDf[foldDf['foldGroup'] <= i].index))
        testRows.append(list(foldDf[foldDf['foldGroup'] == i+1].index))
    
    
    foldIdx = (trainRows, testRows)

    
    return(x_train, y_train, foldIdx)




# trainWindowerRolling: returns the rolling training windows as described in section 3.3 and crossvalidation fold data, using coulumn names to represent sequential data
    # Inputs:
        # trainingDfDict: training data from timeSeriesPercentTrainTestSplit as a dict of dataframes
        # numberOfFolds: number of folds for sequential cross validation (section 3.3 and 4.4). will create numberOfFolds + 1 pools for numberOfFolds sequential evaluations
        # windowLength: input window length
        # dynamicCols: dynamic column names list as created by colTyper
        # cataCols: static column names list as created by colTyper
        # predictionVariable: name of column containing the label binary values
    # Returns:
        # dataframe of training windows, with columns named: name_lag_0, name_lag_1 ....  with the number representing how many iterations back in time that variable represents
        # np.array of the labels corresponding to each window in the dataframe
        # array of arrays to be used by the custom cross validation function, containg the DF index for each cross validations folds
        
# NB: this function needs to be optimised, it currently is extermely slow.
# NB: This is the datafram/flat version as used by all non iterative models such as ours.

def trainWindowerRolling(trainingDfDict, numberOfFolds, windowLength, dynamicCols, cataCols, predictionVariable):
    
    dataFrameList = list()
    labelList = list()
    
    for each in trainingDfDict:
        
        tdf, tl = windower(trainingDfDict[each], windowLength, dynamicCols, cataCols, predictionVariable)
        
        b = np.array_split((np.array(tdf.index.to_series())), numberOfFolds )
        
        foldNumberList = list()
        for i in range(len(b)):
            for each in b[i]:
                foldNumberList.append(i)
                
        tdf['foldGroup'] = foldNumberList 
        
        dataFrameList.append(tdf)
        labelList.append(tl)
        
    
    x_train = pd.concat(dataFrameList)
    x_train.reset_index(inplace = True, drop = True)
    
    y_train  = [j for i in labelList for j in i]
    
    
    trainRows = list()
    testRows = list()
    
    for i in range(0, numberOfFolds - 1):
        trainRows.append(list(x_train[x_train['foldGroup'] <= i].index))
        testRows.append(list(x_train[x_train['foldGroup'] == i+1].index))
        
        
    foldIdx = (trainRows, testRows)
    
    x_train.drop(columns = ['foldGroup'], inplace = True)

    return(x_train, y_train, foldIdx)



# custom_dix_folder_rolling: custom sequential moving block bootstrapping function
    # Inputs:
        # foldIdxs
    # returns:
        # yeilds the index of the train and valids iterativly for each cross validation fold

def custom_idx_folder_rolling(foldIdxs):
    
    numFolds = len(foldIdxs[0])
    
    for a in range(numFolds):

        trains = foldIdxs[0][a]
        tests = foldIdxs[1][a]
        
        yield(np.array(trains),np.array(tests))

        
        
        
        
# testWindower: returns the rolling trainign windows as described in section 3.3
    # Inputs:
        # testingDfDict: testing data from timeSeriesPercentTrainTestSplit as a dict of dataframes
        # windowLength: input window length
        # dynamicCols: dynamic column names list as created by colTyper
        # cataCols: static column names list as created by colTyper
        # predictionVariable: name of column containing the label binary values
    # Returns:
        # dataframe of testing windows, with columns named: name_lag_0, name_lag_1 ....  with the number representing how many iterations back in time that variable represents
        # np.array of the labels corresponding to each window in the dataframe
        
# NB: this function needs to be optimised, it currently is extermely slow.

def testWindower(testingDfDict, windowLength, dynamicCols, cataCols, predictionVariable):
    listOfDfs = list()
    listOfLabels = list()

    for ID in testingDfDict:
        tdf,ldf = windower(testingDfDict[ID], windowLength, dynamicCols, cataCols, predictionVariable)
        listOfDfs.append(tdf)
        listOfLabels.append(ldf)

    x_test = (pd.concat(listOfDfs)).reset_index(drop = True)
    y_test = [j for i in listOfLabels for j in i]
    
    return(x_test, y_test)




# arrayTestWindower: returns the rolling testing windows as described in section 3.3 and crossvalidation fold data, using arrays to represent sequential windows
    # Inputs:
        # testingDfDict: testing data from timeSeriesPercentTrainTestSplit as a dict of dataframes
        # windowLength: input window length
        # predictionVariable: name of column containing the label binary values
        # timeFirst: indication for if window arrays should be time or feature first
            # timeFirst = True: output.shape = (window_number, timestep, feature)
            # timeFirst = False: output.shape = (window_number, feature, timestep)
            # True is used by RNN-BOF
        # sliding: indication for if windows should be sliding or non-overlapping
            # True is used by RNN-BOF
        # Univariate: Use True if data is univariate
        # removeNaNY: If true, removes windows and labels if the label is missing
            # redundant due to other pre-processing steps
    # Returns:
        # 3D np.array representing windows, timesteps and features
        # np.array of the labels corresponding to each window in the dataframe

def arrayTestWindower(testingDfDict, windowLength, predictionVariable, timeFirst, sliding, univariate, removeNaNY):
    listOfMultiXTest = list()
    listOfMultiYTest = list()
    for each in testingDfDict: 
        tx, ty = arrayWindower(testingDfDict[each], windowLength, predictionVariable, timeFirst, sliding , univariate, removeNaNY)
        listOfMultiXTest.append(tx)
        listOfMultiYTest.append(ty)   
        
    gmxTest = np.concatenate(listOfMultiXTest, axis = 0)
    gmyTest = np.concatenate(listOfMultiYTest, axis = 0)

    return(gmxTest, gmyTest)




