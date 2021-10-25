#- MAE - RMSE - R2 - Precision - Recall - F1-Score - Accuracy - Correlation Analysis - MAPE - T-Test
''' !!!
Use micro-averaging score when there is a need to weight each instance or prediction equally.

Use macro-averaging score when all classes need to be treated equally to evaluate the overall performance of the classifier with regard to the most frequent class labels.

Use weighted macro-averaging score in case of class imbalances (different number of instances related to different class labels). The weighted macro-average is calculated
by weighting the score of each class label by the number of true instances when calculating the average.

https://quick-adviser.com/why-is-accuracy-precision-and-recall-the-same/#Can_precision_and_recall_be_equal

!!! '''
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1
import csv
##from sklearn.metrics import precision_recall_fscore_support
from math import sqrt

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

data = pd.read_csv("Data-Analysis/SampleData.csv")

#newdata_cols = ['date', 'actual' , 'predicted']
#newdata = pd.DataFrame(data[['date', 'actual', 'predicted']].values, columns = newdata_cols)

actualPrice = pd.DataFrame(data[['actual']].values)
predictedPrice = pd.DataFrame(data[['predicted']].values)

print("=====>\tMAE")
print(mae(actualPrice, predictedPrice))

print("\n=====>\tRMSE")
print(sqrt(mse(actualPrice, predictedPrice)))

print("\n=====>\tR-Squared")
print(r2(actualPrice, predictedPrice))

print("\n=====>\tMAPE")
print(mape(actualPrice, predictedPrice))

print("\n=====>\Paired T-Test")
print(ttest_rel(actualPrice, predictedPrice, nan_policy='omit'))

dataCorrelation=list(['actual','open_price','24_high','24_low','google','twitter','reddit'])

print("\n=====>\tCorrelation Analysis")
#print(pearsonr(actualPrice, predictedPrice))
print(data[dataCorrelation].corr())

actualPriceDirection= [0]*49
predictedPriceDirection= [0]*49

currPrice = 0
for index, row in actualPrice.iterrows():
    #print (str(index) + " === " + str(row.values))
    currPrice = row.values
    if (index!=0 and lastPrice<currPrice):
        #print("Increased" + str(lastPrice) + " Last < Current " + str(currPrice))
        actualPriceDirection[index-1] = "Increased"
    elif (index!=0 and lastPrice>currPrice):
        #print("Decreased" + str(lastPrice) + " Last > Current" + str(currPrice))
        actualPriceDirection[index-1] = "Decreased"
    lastPrice = row.values

currPrice = 0
for index, row in predictedPrice.iterrows():
    #print (str(index) + " === " + str(row.values))
    currPrice = row.values
    if (index!=0 and lastPrice<currPrice):
        #print("Increased" + str(lastPrice) + " Last < Current " + str(currPrice))
        predictedPriceDirection[index-1] = "Increased"
    elif (index!=0 and lastPrice>currPrice):
        #print("Decreased" + str(lastPrice) + " Last > Current" + str(currPrice))
        predictedPriceDirection[index-1] = "Decreased"
    lastPrice = row.values

#print (predictedPriceDirection)

print()
print(classification_report(actualPriceDirection, predictedPriceDirection, output_dict = True))
print()
print(classification_report(actualPriceDirection, predictedPriceDirection, output_dict = False))

# ===== MACRO =====
print("\n=====>\tPrecision (Actual - Predicted [Macro])")
print('Precision: %.2f' % precision(actualPriceDirection, predictedPriceDirection, average='macro'))

print("\n=====>\tRecall (Actual - Predicted [Macro])")
print('Recall: %.2f' % recall(actualPriceDirection, predictedPriceDirection, average='macro'))

print("\n=====>\F1-Score (Actual - Predicted [Macro])")
print('F1 Score: %.2f' % f1(actualPriceDirection, predictedPriceDirection, average='macro'))

print("\n=====>\tPrecision (Predicted - Actual [Macro])")
print('Precision: %.2f' % precision(predictedPriceDirection, actualPriceDirection, average='macro'))

print("\n=====>\tRecall (Predicted - Actual [Macro])")
print('Recall: %.2f' % recall(predictedPriceDirection, actualPriceDirection, average='macro'))

print("\n=====>\F1-Score (Predicted - Actual [Macro])")
print('F1 Score: %.2f' % f1(predictedPriceDirection, actualPriceDirection, average='macro'))

# ===== MICRO =====
print("\n=====>\tPrecision (Actual - Predicted [Micro])")
print('Precision: %.2f' % precision(actualPriceDirection, predictedPriceDirection, average='micro'))

print("\n=====>\tRecall (Actual - Predicted [Micro])")
print('Recall: %.2f' % recall(actualPriceDirection, predictedPriceDirection, average='micro'))

print("\n=====>\F1-Score (Actual - Predicted [Micro])")
print('F1 Score: %.2f' % f1(actualPriceDirection, predictedPriceDirection, average='micro'))

print("\n=====>\tPrecision (Predicted - Actual [Micro])")
print('Precision: %.2f' % precision(predictedPriceDirection, actualPriceDirection, average='micro'))

print("\n=====>\tRecall (Predicted - Actual [Micro])")
print('Recall: %.2f' % recall(predictedPriceDirection, actualPriceDirection, average='micro'))

print("\n=====>\F1-Score (Predicted - Actual [Micro])")
print('F1 Score: %.2f' % f1(predictedPriceDirection, actualPriceDirection, average='micro'))

# ===== WEIGHTED =====
print("\n=====>\tPrecision (Actual - Predicted [Weighted])")
print('Precision: %.2f' % precision(actualPriceDirection, predictedPriceDirection, average='weighted'))

print("\n=====>\tRecall (Actual - Predicted [Weighted])")
print('Recall: %.2f' % recall(actualPriceDirection, predictedPriceDirection, average='weighted'))

print("\n=====>\tF1-Score (Actual - Predicted [Weighted])")
print('F1 Score: %.2f' % f1(actualPriceDirection, predictedPriceDirection, average='weighted'))

print("\n=====>\tPrecision (Predicted - Actual [Weighted])")
print('Precision: %.2f' % precision(predictedPriceDirection, actualPriceDirection, average='weighted'))

print("\n=====>\tRecall (Predicted - Actual [Weighted])")
print('Recall: %.2f' % recall(predictedPriceDirection, actualPriceDirection, average='weighted'))

print("\n=====>\tF1-Score (Predicted - Actual [Weighted])")
print('F1 Score: %.2f' % f1(predictedPriceDirection, actualPriceDirection, average='weighted'))

# ===== PER CLASS =====
print("\n=====>\tDecreased - Increased (Precision)")
print(precision(actualPriceDirection, predictedPriceDirection, average=None))

print("\n=====>\tDecreased - Increased (Recall)")
print(recall(actualPriceDirection, predictedPriceDirection, average=None))

print("\n=====>\tDecreased - Increased (F1 Score)")
print(f1(actualPriceDirection, predictedPriceDirection, average=None))


#testchanges

#print(precision_recall_fscore_support(actualPriceDirection, predictedPriceDirection, average=None))

'''
print(precision_recall_fscore_support(predictedPriceDirection, actualPriceDirection, average='weighted'))

#actualPrice = (data[['actual']].values).tolist()
#predictedPrice = (data[['predicted']].values).tolist()

actualPrice =   [17,33]
predictedPrice= [9,41]

print(actualPrice)
print(predictedPrice)

print("\n=====>\tPrecision")
print(precision(actualPrice, predictedPrice, average='macro',zero_division=1))

print("\n=====>\tRecall")
print(recall(actualPrice, predictedPrice, average='macro',zero_division=1))
'''

''' ### SAMPLE ### 
truth =      ["Dog","Not a dog","Dog","Dog",      "Dog", "Not a dog", "Not a dog", "Dog",       "Dog", "Not a dog"]
prediction = ["Dog","Dog",      "Dog","Not a dog","Dog", "Not a dog", "Dog",       "Not a dog", "Dog", "Dog"]
print(classification_report(truth, prediction))
'''


#DATA ANALYSIS TABLE FOR MAE,RMSE,R2,MAPE,TTEST
analysis = [mae(actualPrice, predictedPrice),sqrt(mse(actualPrice, predictedPrice)),r2(actualPrice, predictedPrice),
mape(actualPrice, predictedPrice),ttest_rel(actualPrice, predictedPrice, nan_policy='omit')]
analysis_count = 0

with open('mae-ttest.csv', 'w', newline = '') as csvfile:

    fieldnames = ['Method', 'Analysis']
    thewriter = csv.DictWriter(csvfile, fieldnames = fieldnames)
    thewriter.writeheader()

    for analysis in analysis:
        analysis_count += 1
        thewriter.writerow({'Method':analysis_count, 'Analysis':analysis})

#DATA ANALYSIS FOR DATA CORRELATION
data_cor = [data[dataCorrelation].corr()]
data_cor_count = 0

with open('data-correlation.csv', 'w', newline = '') as csvfile:

    fieldnames = ['Method', 'Analysis']
    thewriter = csv.DictWriter(csvfile, fieldnames = fieldnames)
    thewriter.writeheader()

    for data_cor in data_cor:
        data_cor_count += 1
        thewriter.writerow({'Method':data_cor_count, 'Analysis':data_cor})

#DATA ANALYSIS FOR PREDICTED PRICE DIRECTION
price_direct = [classification_report(actualPriceDirection, predictedPriceDirection, output_dict = True),
                classification_report(actualPriceDirection, predictedPriceDirection, output_dict = False)]
price_direct_count = 0

with open('price-direction.csv', 'w', newline = '') as csvfile:

    fieldnames = ['Method', 'Analysis']
    thewriter = csv.DictWriter(csvfile, fieldnames = fieldnames)
    thewriter.writeheader()

    for price_direct in price_direct:
        price_direct_count += 1
        thewriter.writerow({'Method':price_direct_count, 'Analysis':price_direct})

#DATA ANALYSIS FOR ACTUAL-PREDICTED MACRO
actual_predicted_macro = [precision(actualPriceDirection, predictedPriceDirection, average='macro'),
                          recall(actualPriceDirection, predictedPriceDirection, average='macro'),
                          f1(actualPriceDirection, predictedPriceDirection, average='macro'),
                         ]
actual_predicted_macro_count = 0

with open('actual-predicted-macro.csv', 'w', newline = '') as csvfile:

    fieldnames = ['Method', 'Analysis']
    thewriter = csv.DictWriter(csvfile, fieldnames = fieldnames)
    thewriter.writeheader()

    for actual_predicted_macro in actual_predicted_macro:
        actual_predicted_macro_count += 1
        thewriter.writerow({'Method':actual_predicted_macro_count, 'Analysis':actual_predicted_macro})

#DATA ANALYSIS FOR PREDICTED-ACTUAL MACRO
predicted_actual_macro = [precision(predictedPriceDirection, actualPriceDirection, average='macro'),
                          recall(predictedPriceDirection, actualPriceDirection, average='macro'),
                          f1(predictedPriceDirection, actualPriceDirection, average='macro')
                         ]
predicted_actual_macro_count = 0

with open('predicted_actual-macro.csv', 'w', newline = '') as csvfile:

    fieldnames = ['Method', 'Analysis']
    thewriter = csv.DictWriter(csvfile, fieldnames = fieldnames)
    thewriter.writeheader()

    for predicted_actual_macro in predicted_actual_macro:
        predicted_actual_macro_count += 1
        thewriter.writerow({'Method':predicted_actual_macro_count, 'Analysis':predicted_actual_macro})

#DATA ANALYSIS FOR ACTUAL-PREDICTED MICRO
actual_predicted_micro = [ precision(actualPriceDirection, predictedPriceDirection, average='micro'),
                           recall(actualPriceDirection, predictedPriceDirection, average='micro'),
                           f1(actualPriceDirection, predictedPriceDirection, average='micro')
                         ]
actual_predicted_micro_count = 0

with open('actual-predicted-micro.csv', 'w', newline = '') as csvfile:

    fieldnames = ['Method', 'Analysis']
    thewriter = csv.DictWriter(csvfile, fieldnames = fieldnames)
    thewriter.writeheader()

    for actual_predicted_micro in actual_predicted_micro:
        actual_predicted_micro_count += 1
        thewriter.writerow({'Method':actual_predicted_micro_count, 'Analysis':actual_predicted_micro})

#DATA ANALYSIS FOR PREDICTED-ACTUAL MICRO
predicted_actual_micro = [precision(predictedPriceDirection, actualPriceDirection, average='micro'),
                          recall(predictedPriceDirection, actualPriceDirection, average='micro'),
                          f1(predictedPriceDirection, actualPriceDirection, average='micro')
                         ]
predicted_actual_micro_count = 0

with open('predicted_actual-micro.csv', 'w', newline = '') as csvfile:

    fieldnames = ['Method', 'Analysis']
    thewriter = csv.DictWriter(csvfile, fieldnames = fieldnames)
    thewriter.writeheader()

    for predicted_actual_micro in predicted_actual_micro:
        predicted_actual_micro_count += 1
        thewriter.writerow({'Method':predicted_actual_micro_count, 'Analysis':predicted_actual_micro})

#DATA ANALYSIS FOR ACTUAL-PREDICTED WEIGHTED
actual_predicted_weighted = [precision(actualPriceDirection, predictedPriceDirection, average='weighted'),
                             recall(actualPriceDirection, predictedPriceDirection, average='weighted'),
                             f1(actualPriceDirection, predictedPriceDirection, average='weighted')
                            ]
actual_predicted_weighted_count = 0

with open('actual-predicted-weighted.csv', 'w', newline = '') as csvfile:

    fieldnames = ['Method', 'Analysis']
    thewriter = csv.DictWriter(csvfile, fieldnames = fieldnames)
    thewriter.writeheader()

    for actual_predicted_weighted in actual_predicted_weighted:
        actual_predicted_weighted_count += 1
        thewriter.writerow({'Method':actual_predicted_weighted_count, 'Analysis':actual_predicted_weighted})

#DATA ANALYSIS FOR PREDICTED-ACTUAL WEIGHTED
predicted_actual_weighted = [precision(predictedPriceDirection, actualPriceDirection, average='weighted'),
                             recall(predictedPriceDirection, actualPriceDirection, average='weighted'),
                             f1(predictedPriceDirection, actualPriceDirection, average='weighted')
                            ]
predicted_actual_weighted_count = 0

with open('predicted-actual-weighted.csv', 'w', newline = '') as csvfile:

    fieldnames = ['Method', 'Analysis']
    thewriter = csv.DictWriter(csvfile, fieldnames = fieldnames)
    thewriter.writeheader()

    for predicted_actual_weighted in predicted_actual_weighted:
        predicted_actual_weighted_count += 1
        thewriter.writerow({'Method':predicted_actual_weighted_count, 'Analysis':predicted_actual_weighted})

#DATA ANALYSIS PER CLASS
per_class = [precision(actualPriceDirection, predictedPriceDirection, average=None),
             recall(actualPriceDirection, predictedPriceDirection, average=None),
             f1(actualPriceDirection, predictedPriceDirection, average=None)
            ]
per_class_count = 0

with open('per_class.csv', 'w', newline = '') as csvfile:

    fieldnames = ['Method', 'Analysis']
    thewriter = csv.DictWriter(csvfile, fieldnames = fieldnames)
    thewriter.writeheader()

    for per_class in per_class:
        per_class_count += 1
        thewriter.writerow({'Method':per_class_count, 'Analysis':per_class})