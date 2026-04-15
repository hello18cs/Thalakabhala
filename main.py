1. Problem Statement:
Load a dataset, perform data preprocessing, split the dataset into training and testing sets, train a simple classification model, and evaluate its performance. (Dataset: Iris.csv)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


df = pd.read_csv("Iris.csv")

# Features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]   


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Accuracy:", model.score(X_test, y_test))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
Accuracy: 0.9666666666666667

Classification Report:

              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        11
  versicolor       1.00      0.88      0.93         8
   virginica       0.92      1.00      0.96        11

    accuracy                           0.97        30
   macro avg       0.97      0.96      0.96        30
weighted avg       0.97      0.97      0.97        30




Load a dataset containing missing values, identify the missing data using appropriate technique, train a simple classification model, and evaluate its performance.
(Dataset: Titanic.csv)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("Titanic.csv")

df['Age'].fillna(df['Age'].mean(), inplace=True)

df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", model.score(X_test, y_test))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
Accuracy: 0.8268156424581006

Classification Report:

              precision    recall  f1-score   support

           0       0.87      0.86      0.87       116
           1       0.75      0.76      0.76        63

    accuracy                           0.83       179
   macro avg       0.81      0.81      0.81       179
weighted avg       0.83      0.83      0.83       179

C:\Users\PCU\AppData\Local\Temp\ipykernel_2548\1206880206.py:9: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['Age'].fillna(df['Age'].mean(), inplace=True)




Demonstrate unsupervised learning by applying clustering techniques on unlabeled data and interpreting the patterns formed.
(Dataset: Mall_Customers.csv)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("Mall_Customers.csv")

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

model = KMeans(n_clusters=5)
model.fit(X)

labels = model.labels_

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments")
plt.show()
No description has been provided for this image
Demonstrate unsupervised learning by applying K-Means clustering on the Iris dataset. Use the Elbow Method to determine the optimal number of clusters and visualize the clusters formed.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("Iris.csv")

X = df.iloc[:, :-1]

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

model = KMeans(n_clusters=3)
model.fit(X)

labels = model.labels_

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Iris Clusters")
plt.show()
No description has been provided for this image
No description has been provided for this image



Develop a machine learning model using Linear Regression to predict house prices based on various features such as area, number of rooms, location, and other relevant factors. Perform data preprocessing, train the model, and evaluate its performance.
(Dataset: Housing.csv)

# you can use a simple dataset also (Crete a new) or use provided

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("housing.csv")

# Fix missing values
df.fillna(df.mean(), inplace=True)

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

print("R2 Score:", model.score(X_test, y_test))
R2 Score: 0.6473181634322809



Apply Principal Component Analysis (PCA) to reduce the dimensionality of the dataset and visualize the transformed data.
(Dataset: MNIST.csv)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("mnist.csv")

X = df.drop("label", axis=1)
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=50)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


model = LogisticRegression(max_iter=50, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
Accuracy: 0.9006666666666666

Confusion Matrix:
 [[1978    0    9    8    6   28   11    6   22    3]
 [   1 2292    9   12    1   15    4    6   19    4]
 [  18   28 1825   49   32   16   36   29   55    9]
 [  13   16   48 1857    4  105    7   25   52   15]
 [   8   16   14    3 1853    2   25   12   10  104]
 [  21    8   15   71   20 1602   39   17   79   22]
 [  16    6   22    1   21   22 1957    0   17    1]
 [   6   19   31   12   20    2    0 2025    6   67]
 [  25   50   20   80    8   85   15   12 1722   31]
 [  13   21    8   22   99   11    3   86   21 1803]]

Classification Report:
               precision    recall  f1-score   support

           0       0.94      0.96      0.95      2071
           1       0.93      0.97      0.95      2363
           2       0.91      0.87      0.89      2097
           3       0.88      0.87      0.87      2142
           4       0.90      0.91      0.90      2047
           5       0.85      0.85      0.85      1894
           6       0.93      0.95      0.94      2063
           7       0.91      0.93      0.92      2188
           8       0.86      0.84      0.85      2048
           9       0.88      0.86      0.87      2087

    accuracy                           0.90     21000
   macro avg       0.90      0.90      0.90     21000
weighted avg       0.90      0.90      0.90     21000

C:\Users\PCU\anaconda3\envs\Python\Lib\site-packages\sklearn\linear_model\_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(



Perform feature selection using the correlation-based filter method and analyze the importance of features.
(Dataset: Titanic.csv)

pip install seaborn
Collecting seaborn
  Using cached seaborn-0.13.2-py3-none-any.whl.metadata (5.4 kB)
Requirement already satisfied: numpy!=1.24.0,>=1.20 in c:\users\pcu\anaconda3\envs\python\lib\site-packages (from seaborn) (2.0.2)
Requirement already satisfied: pandas>=1.2 in c:\users\pcu\anaconda3\envs\python\lib\site-packages (from seaborn) (2.2.3)
Requirement already satisfied: matplotlib!=3.6.1,>=3.4 in c:\users\pcu\anaconda3\envs\python\lib\site-packages (from seaborn) (3.10.3)
Requirement already satisfied: contourpy>=1.0.1 in c:\users\pcu\anaconda3\envs\python\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.3.2)
Requirement already satisfied: cycler>=0.10 in c:\users\pcu\anaconda3\envs\python\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in c:\users\pcu\anaconda3\envs\python\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (4.58.0)
Requirement already satisfied: kiwisolver>=1.3.1 in c:\users\pcu\anaconda3\envs\python\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.4.8)
Requirement already satisfied: packaging>=20.0 in c:\users\pcu\anaconda3\envs\python\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (24.1)
Requirement already satisfied: pillow>=8 in c:\users\pcu\anaconda3\envs\python\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (11.2.1)
Requirement already satisfied: pyparsing>=2.3.1 in c:\users\pcu\anaconda3\envs\python\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (3.2.3)
Requirement already satisfied: python-dateutil>=2.7 in c:\users\pcu\anaconda3\envs\python\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in c:\users\pcu\anaconda3\envs\python\lib\site-packages (from pandas>=1.2->seaborn) (2024.1)
Requirement already satisfied: tzdata>=2022.7 in c:\users\pcu\anaconda3\envs\python\lib\site-packages (from pandas>=1.2->seaborn) (2025.2)
Requirement already satisfied: six>=1.5 in c:\users\pcu\anaconda3\envs\python\lib\site-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.4->seaborn) (1.16.0)
Using cached seaborn-0.13.2-py3-none-any.whl (294 kB)
Installing collected packages: seaborn
Successfully installed seaborn-0.13.2
Note: you may need to restart the kernel to use updated packages.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Titanic.csv")

df = df.select_dtypes(include='number')
df.fillna(df.mean(), inplace=True)

# Compute correlation matrix
corr_matrix = df.corr()

# Correlation with target (Survived)
target_corr = corr_matrix["Survived"].sort_values(ascending=False)

print("Correlation with Target:\n")
print(target_corr)

# Visualization
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
Correlation with Target:

Survived       1.000000
Fare           0.257307
Parch          0.081629
PassengerId   -0.005007
SibSp         -0.035322
Age           -0.069809
Pclass        -0.338481
Name: Survived, dtype: float64
No description has been provided for this image



Implement standardization (Z-score normalization) and study its effect on model performance.
(Dataset: wine.csv)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("wine.csv")

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model1 = LogisticRegression(max_iter=50, random_state=42)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
acc1 = accuracy_score(y_test, y_pred1)

# -----------------------------
# 2. WITH STANDARDIZATION
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model2 = LogisticRegression(max_iter=50, random_state=42)
model2.fit(X_train_scaled, y_train)
y_pred2 = model2.predict(X_test_scaled)
acc2 = accuracy_score(y_test, y_pred2)

print("Accuracy WITHOUT Scaling:", acc1)
print("Accuracy WITH Scaling:", acc2)
Accuracy WITHOUT Scaling: 0.9444444444444444
Accuracy WITH Scaling: 1.0
C:\Users\PCU\anaconda3\envs\Python\Lib\site-packages\sklearn\linear_model\_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(



Apply Min-Max scaling on the dataset and compare the model performance before and after scaling.
(Dataset: wine.csv)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("wine.csv")


X = df.drop("Class", axis=1)
y = df["Class"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model1 = LogisticRegression(max_iter=50, random_state=42)
model1.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)
acc1 = accuracy_score(y_test, y_pred1)


scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model2 = LogisticRegression(max_iter=50, random_state=42)
model2.fit(X_train_scaled, y_train)

y_pred2 = model2.predict(X_test_scaled)
acc2 = accuracy_score(y_test, y_pred2)

print("Accuracy WITHOUT Scaling:", acc1)
print("Accuracy WITH Min-Max Scaling:", acc2)
Accuracy WITHOUT Scaling: 0.9444444444444444
Accuracy WITH Min-Max Scaling: 1.0
C:\Users\PCU\anaconda3\envs\Python\Lib\site-packages\sklearn\linear_model\_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(


Evaluate the performance of a classification model using confusion matrix, accuracy, precision, recall, and F1-score.
(Dataset: Iris.csv)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,classification_report


df = pd.read_csv("Iris.csv")

X = df.drop("species", axis=1)
y = df["species"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print("Confusion Matrix:\n", cm)
print("\nAccuracy:", acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
Confusion Matrix:
 [[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]

Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1 Score: 1.0

Classification Report:

              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30



Evaluate the performance of a classification model using confusion matrix, accuracy, precision, recall, and F1-score.
(Dataset: diabetes.csv)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
Confusion Matrix:
 [[77 22]
 [21 34]]
Accuracy: 0.7207792207792207
Precision: 0.6071428571428571
Recall: 0.6181818181818182
F1 Score: 0.6126126126126126



Evaluate clustering results using Silhouette Score and Davies-Bouldin Index.
(Dataset: Mall_Customers.csv)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("Mall_Customers.csv")

X = df.drop("CustomerID",  axis=1)
X = df.drop("Gender", axis=1)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="viridis")

plt.title("K-Means Clusters")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

S = silhouette_score(X_scaled, labels)
DB = davies_bouldin_score(X_scaled, labels)

print("Silhouette Score:", S)
print("Davies-Bouldin Index:", DB)
No description has been provided for this image
Silhouette Score: 0.4272395443393026
Davies-Bouldin Index: 0.8262907930832972



13. Evaluate clustering results using Silhouette Score and Davies-Bouldin Index.
(Dataset: diabetes.csv) 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("diabetes.csv")

X = df[["Glucose", "BMI"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Plot clusters
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="viridis")

plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("K-Means Clustering")
plt.colorbar(label="Cluster")
plt.show()

S= silhouette_score(X_scaled, cluster_labels)
DB= davies_bouldin_score(X_scaled, cluster_labels)

print(f"Silhouette Score:", S)
print(f"Davies-Bouldin Index:", DB)
No description has been provided for this image
Silhouette Score: 0.1622463800975957
Davies-Bouldin Index: 1.759873358872424


Implement a Random Forest classifier and compare its performance with a Decision Tree classifier.
(Dataset: Titanic.csv)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("Titanic.csv")

df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]]

df["Age"] = df["Age"].fillna(df["Age"].median())

# Encode categorical data
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

X = df.drop("Survived", axis=1)
y = df["Survived"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
Decision Tree Accuracy: 0.7486033519553073
Random Forest Accuracy: 0.8044692737430168



Implement an XGBoost classifier and compare its performance with a Decision Tree classifier.
(Dataset: Titanic.csv)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

df = pd.read_csv("Titanic.csv")

df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]]
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
Decision Tree Accuracy: 0.7486033519553073
XGBoost Accuracy: 0.8044692737430168
C:\Users\PCU\anaconda3\envs\Python\Lib\site-packages\xgboost\training.py:183: UserWarning: [12:54:14] WARNING: C:\actions-runner\_work\xgboost\xgboost\src\learner.cc:738: 
Parameters: { "use_label_encoder" } are not used.

  bst.update(dtrain, iteration=i, fobj=obj)



Implement a Random Forest classifier and compare its performance with a Decision Tree classifier.
(Dataset: Wine.csv)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("wine.csv")


X = df.drop("Class", axis=1)
y = df["Class"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
Decision Tree Accuracy: 0.9444444444444444
Random Forest Accuracy: 1.0



Implement an XGBoost classifier and compare its performance with a Decision Tree classifier.
(Dataset: Titanic.csv)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

df = pd.read_csv("wine.csv")

X = df.drop("Class", axis=1)
y = df["Class"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

from sklearn.preprocessing import LabelEncoder


xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
Decision Tree Accuracy: 0.9629629629629629
XGBoost Accuracy: 0.9629629629629629



Apply Recursive Feature Elimination (RFE) to select the most relevant features for model building.
(Dataset: Titanic.csv)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


df = pd.read_csv("Titanic.csv")


df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]]


df["Age"] = df["Age"].fillna(df["Age"].median())

df["Sex"] = LabelEncoder().fit_transform(df["Sex"])


X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=1000)

# RFE (select top 2 features)
rfe = RFE(estimator=model, n_features_to_select=2)
rfe.fit(X_train, y_train)

# Selected features
selected_features = X.columns[rfe.support_]

print("Selected Features:", list(selected_features))
Selected Features: ['Pclass', 'Sex']
 