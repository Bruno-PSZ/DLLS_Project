# Packages
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pickle
import numpy as np
import pandas as pd


def evaluate(model, X_test, y_test, label_encoder=None):
    # Global metrics: General accuracy and macro_F1:
    y_pred = model.predict(X_test)
    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred)
    ACC = accuracy_score(y_test, y_pred, normalize=True)
    macro_F1 = f1_score(y_test, y_pred, average='macro')

    with open("./util/cell_types.pkl", "rb") as f:
        cell_types = pickle.load(f)
        
    # Confusion matrix:
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation='vertical', display_labels=cell_types)
    fig = disp.figure_
    fig.set_figwidth(11)
    fig.set_figheight(8) 
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.suptitle('Confusion matrix')

    # Accuracy for each class:
    confusion_matrix = disp.confusion_matrix
    ACC_per_celltype = confusion_matrix.diagonal()/confusion_matrix.sum(axis=1)

    # AUC, Average Precision Score (AP) and ACC across cell types
    y_prob = model.predict_proba(X_test)
    AUC_per_celltype = roc_auc_score(y_test, y_prob, average=None, multi_class='ovr', labels=cell_types)
    AP_per_celltype = average_precision_score(y_test, y_prob, average=None)
    res = np.concatenate((AUC_per_celltype.reshape(-1, 1), AP_per_celltype.reshape(-1, 1), ACC_per_celltype.reshape(-1, 1)), axis=1)
    res_df = pd.DataFrame(res, columns=["AUC","AP","ACC"], index=cell_types)

    # Sklearn report
    report = classification_report(y_test, y_pred, target_names=cell_types)

    return {"ACC_glob":ACC, "macro_F1":macro_F1, "conf_plot":fig, "results_per_celltypes":res_df, "sklearn_report":report}
    