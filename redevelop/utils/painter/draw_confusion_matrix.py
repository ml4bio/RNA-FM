# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:44:51 2019

@author: admin
"""

#print(__doc__)

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")    # matplotlib运行出现的Invalid DISPLAY variable, 切换后端
import matplotlib.pyplot as plt


# self-made for input of numpy
def plot_confusion_matrix(confusion_matrix,
                          classes,
                          normalize=False,
                          cmap=plt.cm.Blues,
                          save_file=None,
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param confusion_matrix: input numpy  (true, pred)
    :param classes: list of class label (for x and y tick labels)
    :param save_file:
    :param retrun: canvas_img, the visual map of CM plotted by matplotlib, which is a numpy for direct display (eg. in summaryWritter)
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    else:
        confusion_matrix = confusion_matrix.astype('int')

    num_class = len(classes)

    fig, ax = plt.subplots()
    fig.set_size_inches(num_class * 2, num_class * 2)

    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    fig.tight_layout()

    # fig 转为 numpy
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    if save_file is not None:
        save_dir, filename = os.path.split(save_file)
        if os.path.exists(save_dir) != True:
            os.makedirs(save_dir)
        plt.savefig(save_file)

    # plt.show()  # can directly save without showing
    plt.close("all")
    return data  # numpy  （W，H，C）  rgb


# Draw ConfusionMatrix  Another Way Use seaborn
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
a = pd.DataFrame(metrics["confusion_matrix"], columns=classes_list, index=classes_list)
ax = sns.heatmap(a, annot=True)
ax.set_xlabel("Predict label")
ax.set_ylabel("True label")
ax.set_title("Confusion matrix")
plt.savefig("ConfusionMatrix.png", dpi=300)
plt.show()
plt.close()
"""