import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import tree
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn import svm
import copy
from sklearn import datasets
from sklearn.inspection import permutation_importance
import os
from skimage.transform import resize
import cv2
import sklearn.model_selection as model_selection

#Load dataset
botnet_detection_dataset = pd.read_csv('./datasets/Botnet detection dataset.csv')
botnet_detection_dataset.reset_index(drop=True, inplace=True)
botnet_detection_dataset.replace('', np.nan, inplace=True)
botnet_detection_dataset.fillna(0, inplace=True)
print(len(botnet_detection_dataset))
print(botnet_detection_dataset.head())
#split dataset into parts
def datasetSplitCompute(train_dataset, train_dataset_label,  num_parts):
    num_samples = len(train_dataset)
    print(num_samples)
    shuffledDataset = np.random.permutation(num_samples)
    part_size = num_samples / num_parts
    print(part_size)
    part_indices = []
    part_dataset = []
    partitioned_dataset = {}
    for num in range(0, num_parts):
        print(int((num)*part_size),":",int((num+1)*part_size))
        variable_dataset = "dataset_part"+str(num)
        variable_dataset_label = "dataset_part_label"+str(num)
        partitioned_dataset[variable_dataset] = train_dataset[int((num)*part_size):int((num+1)*part_size)]
        partitioned_dataset[variable_dataset_label] = train_dataset_label[int((num)*part_size):int((num+1)*part_size)]
    return partitioned_dataset


def poisoning_label_flipping(dataset, label, poison):
    #taking 12.5% of dataset length
    label = label.values
    print(len(label))
    randomLength = int(len(dataset)*poison)
    #initiate 
    indexHolder = []
    random_integer = 0
    i = 0
    while(len(indexHolder)<randomLength):
        random_integer = random.randint(0, len(dataset)-1)
        if (random_integer not in indexHolder):
            if(label[random_integer] == 0):
                 label[random_integer] = 1
            else:
                label[random_integer] = 0
        indexHolder.append(random_integer)
        i+=1
    poisonedLabels = pd.DataFrame(label)
    poisonedLabels.rename(columns={'0': 'label'}, inplace=True)
    print(poisonedLabels.columns)
    return poisonedLabels



labelEncoder = LabelEncoder()

images = []
labels = []
i=1
X = botnet_detection_dataset.drop(columns=["label"])
y = botnet_detection_dataset[["label"]]

# Define and train a SVM model (binary classification assumed)
SGDClf = svm.SVC()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)

# PCA for dimensionality reduction (choose the number of components)
pca = PCA(n_components=3)  # Adjust the number of components as needed
pca_data = pca.fit_transform(data_scaled)
SGDClf.fit(pca_data, y)
perm_importance = permutation_importance(SGDClf, pca_data, y)

# Making the sum of feature importance being equal to 1.0,
# so feature importance can be understood as percentage
perm_importance_normalized = perm_importance.importances_mean/perm_importance.importances_mean.sum()
print("permutation importance: ", perm_importance_normalized)

support_vectors = SGDClf.support_vectors_  # Get the support vectors
dual_coefficients = SGDClf.dual_coef_

weights = SGDClf.decision_function(pca_data)

# Calculate feature importance scores
feature_importance = np.mean(weights, axis=0)
print(feature_importance)


decision_values = SGDClf.decision_function(pca_data)
margin_score = np.min(np.abs(decision_values))

print("Margin score of SVM with SGD model:", margin_score)

features_importance = SGDClf.dual_coef_
normalized_coefficients = np.abs(features_importance) / np.linalg.norm(features_importance)

# Calculate feature importance scores
feature_importance_scores = np.mean(normalized_coefficients, axis=0)
print(feature_importance_scores)

y_poisoned_10 = poisoning_label_flipping(pca_data, y, 0.10)

X_poisoned = X
SGDClf.fit(pca_data, y_poisoned_10)

perm_importance_10 = permutation_importance(SGDClf, pca_data, y_poisoned_10)

# Making the sum of feature importance being equal to 1.0,
# so feature importance can be understood as percentage
perm_importance_normalized_10 = perm_importance_10.importances_mean/perm_importance_10.importances_mean.sum()
print("permutation importance: ", perm_importance_normalized_10)

decision_values = SGDClf.decision_function(pca_data)
weights_10 = SGDClf.decision_function(pca_data)
margin_score = np.min(np.abs(decision_values))
print("Margin score of SVM with SGD model:", margin_score)

p_features_importance_10 = SGDClf.dual_coef_
normalized_coefficients_10 = np.abs(p_features_importance_10) / np.linalg.norm(p_features_importance_10)
p_feature_importance_scores_10 = np.mean(weights_10, axis=0)
print(p_feature_importance_scores_10)


y_poisoned_15 = poisoning_label_flipping(pca_data, y, 0.15)
SGDClf.fit(pca_data, y_poisoned_15)
perm_importance_15 = permutation_importance(SGDClf, pca_data, y_poisoned_15)

# Making the sum of feature importance being equal to 1.0,
# so feature importance can be understood as percentage
perm_importance_normalized_15 = perm_importance_15.importances_mean/perm_importance_15.importances_mean.sum()
print("permutation importance: ", perm_importance_normalized_15)

weights_15 = SGDClf.decision_function(pca_data)
p_features_importance = SGDClf.dual_coef_
p_normalized_coefficients = np.abs(p_features_importance) / np.linalg.norm(p_features_importance)
p_features_importance_scores = np.mean(weights_15, axis=0)
print(p_features_importance_scores)


margin_score = np.min(np.abs(weights_15))

print("Margin score of SVM with SGD model:", margin_score)
