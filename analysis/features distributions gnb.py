from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def randomSelect(dataset, label, poison):
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
            # print("random integer: ", random_integer)
            # print(label[random_integer])
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



botnet_detection_dataset = pd.read_csv('E:/study/label-flipping attack/Botnet dataset.csv')
botnet_detection_dataset.reset_index(drop=True, inplace=True)
botnet_detection_dataset.replace('', np.nan, inplace=True)
botnet_detection_dataset.fillna(0, inplace=True)
# botnet_detection_dataset.drop(columns=['state', 'proto', 'service', 'attack_cat'], inplace=True)
print(len(botnet_detection_dataset))
print(botnet_detection_dataset.head())
X = botnet_detection_dataset.drop(columns=["label"])
y = botnet_detection_dataset[["label"]]

# Split the dataset into training and testing sets

scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)

# PCA for dimensionality reduction (choose the number of components)
pca = PCA(n_components=3)  # Adjust the number of components as needed
pca_data = pca.fit_transform(data_scaled)

# X_train, X_test, y_train, y_test = train_test_split(pca_data, y, test_size=0.2, random_state=42)

# Initialize and train the Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(pca_data, y)

# Get the mean and variance of each feature for each class
feature_stats = {}
# print(gnb.theta_)
# print(gnb.var_)
for class_label in range(len(gnb.theta_)):
    class_mean = gnb.theta_[class_label]
    class_variance = gnb.var_[class_label]
    probability = gnb.class_prior_[class_label]
    feature_stats[f'Class {class_label}'] = pd.DataFrame({'Mean': class_mean, 'Variance': class_variance, 'Class Probability': probability})

# Display the feature distribution for each class
for class_label, stats_df in feature_stats.items():
    print(f"Class {class_label}:")
    print(stats_df)
    print()
    
gnb_poisoned_labels = randomSelect(X, y, 0.10)

gnb.fit(pca_data, gnb_poisoned_labels)

# Get the mean and variance of each feature for each class
feature_stats = {}
# print(gnb.theta_)
# print(gnb.var_)
for class_label in range(len(gnb.theta_)):
    class_mean = gnb.theta_[class_label]
    class_variance = gnb.var_[class_label]
    probability = gnb.class_prior_[class_label]
    feature_stats[f'Class {class_label}'] = pd.DataFrame({'Mean': class_mean, 'Variance': class_variance, 'Class Probability': probability})

# Display the feature distribution for each class
for class_label, stats_df in feature_stats.items():
    print(f"Class {class_label}:")
    print(stats_df)
    print()
    
gnb_poisoned_labels = randomSelect(X, y, 0.15)

gnb.fit(pca_data, gnb_poisoned_labels)

# Get the mean and variance of each feature for each class
feature_stats = {}
# print(gnb.theta_)
# print(gnb.var_)
for class_label in range(len(gnb.theta_)):
    class_mean = gnb.theta_[class_label]
    class_variance = gnb.var_[class_label]
    probability = gnb.class_prior_[class_label]
    feature_stats[f'Class {class_label}'] = pd.DataFrame({'Mean': class_mean, 'Variance': class_variance, 'Class Probability': probability})

# Display the feature distribution for each class
for class_label, stats_df in feature_stats.items():
    print(f"Class {class_label}:")
    print(stats_df)
    print()