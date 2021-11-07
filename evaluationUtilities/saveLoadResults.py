import pickle
import os




def resultsSave(modelName, bParams, trueValues, probaPred, history, tune_time, ftrain_time):
    
    saveName = modelName
    
    modelName = dict()

    modelName['model_params'] = bParams

    modelName['model_proba'] = (trueValues, probaPred)
    
    modelName['tf_history'] = history
    
    modelName['tune_train_time'] = (tune_time, ftrain_time)

    filename = 'Results/'+saveName
    
    with open(filename, 'wb')as fp:
        
        pickle.dump(modelName, fp)
        
        
        
def resultsLoad(directory):
    
    resultsDict = dict()

    for filename in os.listdir(directory):
        if filename != '.ipynb_checkpoints':
            with open(directory + '/' + filename, 'rb') as fr:
                a = pickle.load(fr)
                resultsDict[filename] = a