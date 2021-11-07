import pandas as pd
import numpy as np
import datetime
import scipy
import sklearn 
from prg import prg
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import average_precision_score


def auprg(y_true, y_pred):
    prg_curve = prg.create_prg_curve(y_true, y_pred)
    return(prg.calc_auprg(prg_curve))


def prgCurveMetric(y_true, y_pred):
    prg_curve = prg.create_prg_curve(y_true, y_pred)
    print('AUPRG:',prg.calc_auprg(prg_curve))
    prg.plot_prg(prg_curve)
    return()


