import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
import random
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
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

# Generate synthetic dataset
botnet_detection_dataset = pd.read_csv('E:/study/label-flipping attack/Botnet dataset.csv')
botnet_detection_dataset.reset_index(drop=True, inplace=True)
botnet_detection_dataset.replace('', np.nan, inplace=True)
botnet_detection_dataset.fillna(0, inplace=True)
# botnet_detection_dataset.drop(columns=['state', 'proto', 'service', 'attack_cat'], inplace=True)
print(len(botnet_detection_dataset))
print(botnet_detection_dataset.head())
X = botnet_detection_dataset.drop(columns=["label"])
pca = PCA(n_components=2)  # Adjust the number of components as needed
X = pca.fit_transform(X)
# X = X.iloc[:, :3]
y = botnet_detection_dataset[["label"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train KNN classifier on original dataset
knn_original = KNeighborsClassifier(n_neighbors=5)
knn_original.fit(X_train, y_train)

y_poisoned = randomSelect(X_train, y_train, 0.15)
X_poisoned = X_train


# Train KNN classifier on poisoned dataset
# Assume 'X_poisoned' and 'y_poisoned' are the poisoned dataset
knn_poisoned = KNeighborsClassifier(n_neighbors=5)
knn_poisoned.fit(X_poisoned, y_poisoned)

# Create mesh grid
h = 0.05
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict class labels for mesh grid points
Z_original = knn_original.predict(np.c_[xx.ravel(), yy.ravel()])
Z_poisoned = knn_poisoned.predict(np.c_[xx.ravel(), yy.ravel()])

# Reshape predictions to match mesh grid shape
Z_original = Z_original.reshape(xx.shape)
Z_poisoned = Z_poisoned.reshape(xx.shape)

mapping = {0: 'benign', 1: 'malignant'}

# Example usage:
# Map values to labels
value_to_label = [mapping[value] for value in y_train['label']]
# Plot original decision boundaries
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_original, alpha=0.8)
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1],
palette=['orange', 'c', 'blue'], hue=value_to_label, alpha=1.0, edgecolor="black")
plt.title('kNN Decision Boundaries with Clean Dataset')

# Plot poisoned decision boundaries
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_poisoned, alpha=0.8)
sns.scatterplot(x=X_poisoned[:, 0], y=X_poisoned[:, 1],
palette=['orange', 'c', 'blue'], hue=value_to_label, alpha=1.0, edgecolor="black")
plt.title('kNN Decision Boundaries with \u0394P=15.0% Dataset Poisoning')

plt.show()
