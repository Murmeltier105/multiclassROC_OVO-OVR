import glob
import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
from itertools import cycle

# setup lists of classifications you want to compare
root = '/home/dr1/PycharmProjects/GraMa/'
classifications = ['trainedModelsNormalized/Ce2No', 'trainedModelsNormalized/Ce2NoGroupAug+', 'trainedModelsNormalized/Bo2No', 'trainedModelsNormalized/Bo2NoGroupAug+']
#the folder root + classification must contain a file '*Vorhersagewerte3*.pkl' with the predictive values at columns [3:6]
#other neccesary columns: 'Label', 'imagePath'
save = False    #set to True if you like the plot
# ## Functions
# Slightly modified from the binary classifier case


def calculate_tpr_fpr(y_real, y_pred):
    '''
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations

    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes

    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    '''

    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # Calculates tpr and fpr
    tpr = TP / (TP + FN)  # sensitivity - true positive rate
    fpr = 1 - TN / (TN + FP)  # 1-specificity - false positive rate

    return tpr, fpr

def get_all_roc_coordinates(y_real, y_proba):
    '''
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a treshold for the predicion of the class.

    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.

    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    '''
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list


def plot_roc_curve(tpr, fpr, name, auc, scatter=True, ax=None):
    '''
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).

    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        name: (str) name of the classification to be displayed in the label
        auc:  (float) AUC value to be displayed alongside the classification inside the label
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
    '''
    if ax == None:
        plt.figure(figsize=(5, 5))
        ax = plt.axes()

    if scatter:
        sns.scatterplot(x=fpr, y=tpr, ax=ax)
    sns.lineplot(x=fpr, y=tpr, ax=ax, legend= 'brief', label = f'{name} (AUC={auc:0.3f})', alpha = 0.8)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    #ax.legend()


# ## Multiclass classification evaluation with KS test

# ### Creating a synthetic dataset


dateStamp = time.strftime("%Y-%m-%d")   #to distinguish png plots
fig, ax = plt.subplots(figsize=(10, 6.5), dpi=400)
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
colors = cycle(["aqua", "darkorange", "cornflowerblue", 'lime'])
ax.spines[['right', 'top']].set_visible(False)
date = time.strftime("%Y-%m-%d")

for classification,color in zip(classifications, colors):
    #load dataframe
    dfPath = glob.glob(f'{root}{classification}/*Vorhersagewerte3*.pkl')
    df = pd.read_pickle(dfPath[0])

#fig = plt.figure(figsize = (6.4,4.8), dpi = 400)
#ax = fig.add_axes([0, 0, 1, 1])


# ## ROC Curve - One vs Rest (OvR)
# Compares each class with the rest of the classes

    #set up variables
    y_proba = np.asarray(df.iloc[:, 3:6])
    #y_pred = np.asarray(df.iloc[:, 2])
    X_test = df[['imagePath']]
    y_test = df['Label'].squeeze()

    # classes = model_multiclass.classes_
    classes = np.asarray([0])

    roc_auc_ovr = {}


#     roc_auc_ovr[c] = roc_auc_score(df_aux['class'], df_aux['prob'])


#%% plot all OvR in one plot


    #fig, ax = plt.subplots(figsize=(6, 6))


    c = 0   #classes[class_id]

    # Prepares an auxiliar dataframe to help with the plots
    df_aux = X_test.copy()
    df_aux['class'] = [1 if y == c else 0 for y in y_test]
    df_aux['prob'] = y_proba[:, 0]
    df_aux = df_aux.reset_index(drop=True)

    # Calculates the ROC AUC OvR
    roc_auc_ovr[c] = roc_auc_score(df_aux['class'], df_aux['prob'])
    tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])

    #PLOTTING
    plot_roc_curve(tpr, fpr,
                   os.path.basename(classification).replace('Ce','ZT').replace('Bo','PT').replace('Aug+',''),   #name in label
                   roc_auc_ovr[c],  #AUC value for legend entry
                   scatter=False, ax=ax)
    #ax.legend(f'ROC {c} vs Rest (AUC={roc_auc_ovr[c]})')
    #ax_bottom.set_title(f"ROC Curve {c} vs. rest")




        #RocCurveDisplay.from_predictions(
         #   y_onehot_test[:, class_id],
          #  y_score[:, class_id],
           # name=f"ROC curve for {target_names[class_id]}",
            #color=color,
            #ax=ax,
        #)

#Layout of the plot
plt.plot([0, 1], [0, 1], "k--", label="Zufall (AUC = 0.5)") #ads diagonal line
plt.axis("square")
plt.xlabel("Falsch positiv Rate")
plt.ylabel("Richtig positiv Rate")
plt.title("Receiver Operating Characteristics\n N- vs. N+")
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(bbox_to_anchor=(0.5, 0.5))

plt.tight_layout()
if save_
    plt.savefig(f'{date}-OvR-N0vsN+.png')
plt.show()




#
#
# #%% Displays the ROC AUC for each class
# avg_roc_auc = 0
# i = 0
# for k in roc_auc_ovr:
#     avg_roc_auc += roc_auc_ovr[k]
#     i += 1
#     print(f"{k} ROC AUC OvR: {roc_auc_ovr[k]:.4f}")
# print(f"average ROC AUC OvR: {avg_roc_auc / i:.4f}")
#
# # Compares with sklearn (average only)
# # "Macro" average = unweighted mean
# roc_auc_score(y_test, y_proba, labels=classes, multi_class='ovr', average='macro')
#
# #%% ROC Curve - One vs One (OvO)
# # Compares each possible combination of the classes, two at a time
# classes_combinations = []
# class_list = list(classes)
# for i in range(len(class_list)):
#     for j in range(i + 1, len(class_list)):
#         classes_combinations.append([class_list[i], class_list[j]])
#         classes_combinations.append([class_list[j], class_list[i]])
# classes_combinations
#
#
#
#
# # Plots the Probability Distributions and the ROC Curves One vs ONe
# plt.figure(figsize=(20,7),dpi=500)
# bins = [i / 20 for i in range(20)] + [1]
# roc_auc_ovo = {}
#
# for i in range(len(classes_combinations)):
#     # Gets the class
#     comb = classes_combinations[i]
#     c1 = comb[0]
#     c2 = comb[1]
#     c1_index = class_list.index(c1)
#     title = str(c1) + " vs " + str(c2)
#
#     # Prepares an auxiliar dataframe to help with the plots
#     df_aux = X_test.copy()
#     df_aux['class'] = y_test
#     df_aux['prob'] = y_proba[:, c1_index]
#
#     # Slices only the subset with both classes
#     df_aux = df_aux[(df_aux['class'] == c1) | (df_aux['class'] == c2)]
#     df_aux['class'] = [1 if y == c1 else 0 for y in df_aux['class']]
#     df_aux = df_aux.reset_index(drop=True)
#
#     # Plots the probability distribution for the class and the rest
#     ax = plt.subplot(2,6, i + 1)
#     sns.histplot(x="prob", data=df_aux, hue='class', color='b', ax=ax, bins=bins)
#     ax.set_title(title)
#     ax.legend([f"Class {c1}", f"Class {c2}"])
#     ax.set_xlabel(f"P(x = {c1})")
#
#     # Calculates the ROC Coordinates and plots the ROC Curves
#     ax_bottom = plt.subplot(2,6, i + 7)
#     tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
#     plot_roc_curve(tpr, fpr, scatter=False, ax=ax_bottom)
#     ax_bottom.set_title(f"ROC Curve {c1}v{c2}")
#
#     # Calculates the ROC AUC OvO
#     roc_auc_ovo[title] = roc_auc_score(df_aux['class'], df_aux['prob'])
#
# plt.tight_layout()
# plt.savefig(f'{date}_histplot-OvO.png')
# plt.show()
# # In[24]:


# # Displays the ROC AUC for each class
# avg_roc_auc = 0
# i = 0
# for k in roc_auc_ovo:
#     avg_roc_auc += roc_auc_ovo[k]
#     i += 1
#     print(f"{k} ROC AUC OvO: {roc_auc_ovo[k]:.4f}")
# print(f"average ROC AUC OvO: {avg_roc_auc / i:.4f}")
#
# # In[25]:
#
#
# # Compares with sklearn (average only)
# # "Macro" average = unweighted mean
# skAverage = roc_auc_score(y_test, y_proba, labels=classes, multi_class='ovo', average='macro')
# print(f'average ROC AUC by sklearn: {skAverage}')
# # In[ ]:
#



