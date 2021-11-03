import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from math import sqrt

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

#For MAE - RMSE - R-Squared - MAPE
def error_analysis(crypto_df):
    error_analysis_df = pd.DataFrame(columns = ['MAE','RMSE','R-Squared','Mape'])
    actualPrice = pd.DataFrame(crypto_df[['actual']].values)
    predictedPrice = pd.DataFrame(crypto_df[['predicted']].values)
    mae_res = mae(actualPrice, predictedPrice)
    rmse_res = sqrt(mse(actualPrice, predictedPrice))
    r2_res = r2(actualPrice, predictedPrice)
    mape_res = mape(actualPrice, predictedPrice)
    pd_data = pd.Series([mae_res , rmse_res, r2_res , mape_res], index=error_analysis_df.columns)
    error_analysis_df = error_analysis_df.append(pd_data,ignore_index=True)
    return error_analysis_df


#For Correlation Analysis
def corr_analysis(crypto_df):
    dataCorrelation=list(['actual','open','24_high','24_low','google','twitter','reddit'])
    return crypto_df[dataCorrelation].corr(method = 'pearson')


# For Precision - Recall - F1-Score - Accuracy
def classification_analysis(crypto_df):
    classification_analysis_df = pd.DataFrame(columns = ['Precision','Recall','F1-Score','Accuracy'])
    actualPrice = pd.DataFrame(crypto_df[['actual']].values)
    predictedPrice = pd.DataFrame(crypto_df[['predicted']].values)
    actualPriceDirection= [0]*(len(crypto_df)-1)
    predictedPriceDirection= [0]*(len(crypto_df)-1)
    currPrice = 0
    for index, row in actualPrice.iterrows():
        currPrice = row.values
        if (index!=0 and lastPrice<currPrice):
            actualPriceDirection[index-1] = 1
        elif (index!=0 and lastPrice>currPrice):
            actualPriceDirection[index-1] = 0
        lastPrice = row.values

    currPrice = 0
    for index, row in predictedPrice.iterrows():
        currPrice = row.values
        if (index!=0 and lastPrice<currPrice):
            predictedPriceDirection[index-1] = 1
        elif (index!=0 and lastPrice>currPrice):
            predictedPriceDirection[index-1] = 0
        lastPrice = row.values

    prec_sco = precision(actualPriceDirection,predictedPriceDirection)
    rec_sco = recall(actualPriceDirection,predictedPriceDirection)
    f1_sco = f1(actualPriceDirection,predictedPriceDirection)
    acc_sco = accuracy(actualPriceDirection,predictedPriceDirection)
    #print(actualPriceDirection)
    #print(predictedPriceDirection)

    #Print Confusion Matrix and Plot  
    #cmtx = pd.DataFrame(confusion_matrix(actualPriceDirection, predictedPriceDirection, labels=['0','1']), index=['true:0', 'true:1'], columns=['pred:0', 'pred:1'])
    cm = confusion_matrix(actualPriceDirection, predictedPriceDirection)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Decreased','Increased']
    plt.title('Price Direction (Increased or Decreased)')
    plt.ylabel('Actual Price')
    plt.xlabel('Predicted Price')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()
    pd_data = pd.Series([prec_sco , rec_sco, f1_sco , acc_sco], index=classification_analysis_df.columns)
    classification_analysis_df = classification_analysis_df.append(pd_data,ignore_index=True)
    return classification_analysis_df
        

BTC_data = pd.read_csv("Data-Analysis/BTC_Sample.csv")
ETH_data = pd.read_csv("Data-Analysis/ETH_Sample.csv")
DOGE_data = pd.read_csv("Data-Analysis/DOGE_Sample.csv")

print(corr_analysis(BTC_data))
print(error_analysis(BTC_data))
print(classification_analysis(BTC_data))
