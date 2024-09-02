import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets
import sklearn.model_selection as model_selection
from sklearn.model_selection import learning_curve
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from skimage.transform import resize
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random

variances = []
variances_sgd = []
variances_dtree = []
variances_knn = []
variances_rforest = []
variances_gnb = []
variances_perceptron = []
variances_mlp = []
#Load dataset
botnet_detection_dataset = pd.read_csv('./datasets/UNSW_NB15 dataset.csv')
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
# Split dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = model_selection.train_test_split(botnet_detection_dataset.drop(columns=["label"]), 
                                                    botnet_detection_dataset[["label"]], 
                                                    test_size=0.3,random_state=109) 
true_labels = np.array(y_test)
true_labels = labelEncoder.fit_transform(true_labels)

poisonLevels = [0.00, 0.05, 0.10, 0.15, 0.2, 0.25]
for level in poisonLevels:
    poisonedLabels = randomSelect(X_train, y_train, level)

    # Train SDG model
    sgd_clean_model_poly = SGDClassifier()
    sgd_clean_model_poly.fit(X_train, y_train)

    # Calculate variance of decision function
    encodedPredictions_sgd = labelEncoder.fit_transform(sgd_clean_model_poly.predict(X_test))
    variance_sgd = np.var(encodedPredictions_sgd)
    variances_sgd.append(variance_sgd) 

    # Train SVM model
    dtree_clean_model = tree.DecisionTreeClassifier()
    dtree_clean_model.fit(X_train, y_train)

    # Calculate variance of decision function
    predictions_dtree = dtree_clean_model.predict(X_test)
    print(predictions_dtree)
    encodedPredictions = labelEncoder.fit_transform(predictions_dtree)
    print(type(predictions_dtree))
    variance_dtree = np.var(np.array(encodedPredictions))
    variances_dtree.append(variance_dtree)
    
    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(X_train, y_train)
    pred_knn = knn_classifier.predict(X_test)
    encodedPredictions_knn = labelEncoder.fit_transform(pred_knn)

    # Calculate variance of decision function
    pred_knn_mean = np.mean(encodedPredictions_knn, axis=0)
    # predicted_mean = np.mean(predicted_labels, axis=0)
    
    # Calculate the variance
    variance_knn = np.mean((true_labels - pred_knn_mean) ** 2)
    variances_knn.append(variance_knn)
    # y_pred = cross_val_predict(knn_classifier, X_test, y_test, cv=5)

    # Train SVM model
    rforest = RandomForestClassifier(n_estimators=9, criterion="log_loss")
    rforest.fit(X_train, y_train)
    y_pred = rforest.predict(X_test)
    encodedPredictions_rf = labelEncoder.fit_transform(y_pred)
    # Calculate variance of decision function
    variance_rforest = np.var(encodedPredictions_rf)
    variances_rforest.append(variance_rforest)

    # Train SVM model
    gNaiveBayes = GaussianNB()
    gNaiveBayes.fit(X_train, y_train)
    y_pred_gnb = gNaiveBayes.predict(X_test)
    encodedPredictions_gnb = labelEncoder.fit_transform(y_pred_gnb)
    # Calculate variance of decision function
    variance_gnb = np.var(encodedPredictions_gnb)
    variances_gnb.append(variance_gnb)

    # Train SVM model
    perceptron = Perceptron(penalty='elasticnet')
    perceptron.fit(X_train, y_train)
    pred_perceptron = perceptron.predict(X_test)
    encodedPredictions_perceptron = labelEncoder.fit_transform(pred_perceptron)
    predicted_labels = np.array(encodedPredictions_perceptron)
    
    # Calculate the mean of predicted labels
    predicted_mean = np.mean(predicted_labels, axis=0)
    
    # Calculate the variance
    variance_perceptron = np.mean((true_labels - predicted_mean) ** 2)
    print("perceptron variance: ", variance_perceptron)
    variances_perceptron.append(variance_perceptron)

    # Train SVM model
    mlPerceptron = MLPClassifier()
    mlPerceptron.fit(X_train, y_train)
    
# Plot variance as a function of Poison Level
plt.plot(poisonLevels, variances_sgd, marker='o', label='variance(\u03C3) in SVM')
plt.plot(poisonLevels, variances_dtree, marker='o', label='variance(\u03C3) in Decision tree')
plt.plot(poisonLevels, variances_knn, marker='o', label='variance(\u03C3) in KNN')
plt.plot(poisonLevels, variances_rforest, marker='o', label='variance(\u03C3) in Random forest')
plt.plot(poisonLevels, variances_gnb, marker='o', label='variance(\u03C3) in GNB')
plt.plot(poisonLevels, variances_perceptron, marker='o', label='variance(\u03C3) in Perceptron')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel("Poison level(%)")
plt.ylabel('Variance(\u03C3)')

plt.show()
