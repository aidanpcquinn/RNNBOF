import pandas as pd

# manyTimeToDict: Converts a global dataframe containing time stamps and ids into dict of dataframes per entity where keys are the entity IDs.
    # Inputs:
        # sortColumnName: the column to sort on (ascending) 
        # dropID: sepcifies if the id column should be dropped from individual dataframes
        # dataFrame: input global dataframe (contains multiple entities entries over time)
        # idColumn: the column entitiy dataframes are grouped on.
    # Returns:
        # dict of dataframes, with keys representing the entities IDs, ordered by the sorting column

def manyTimeToDict( idColumn, dataFrame,  sortColumnName, dropID = True):
    tdict = dict()
    
    for each in list(dataFrame[idColumn].unique()):
        singleDataFrame = dataFrame.loc[dataFrame[idColumn] == each]
        if sortColumnName:
            singleDataFrame = singleDataFrame.sort_values(sortColumnName)
        if dropID == True:
            singleDataFrame.drop(idColumn, axis = 1, inplace = True)
            
        tdict[each] = singleDataFrame
    
    return(tdict)
    