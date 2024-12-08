import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, recall_score, precision_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import pickle
from sklearn import tree
from sklearn import svm
from sklearn.svm import SVC
# Đánh giá nghi thức K-fold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
# Load data từ file Excel
data = pd.read_excel('Rice2024_v3.xlsx')
# Vẽ Box Plot cho từng cột
# plt.figure(figsize=(10, 6))
# sns.boxplot(data= data['Minor_Axis_Length'])
# plt.show()
# Thông tin về dataset (kiểu dữ liệu, số lượng mẫu, giá trị null)
# data.info()

# print("========================")
# Số lượng mẫu và số lượng thuộc tính
# print("Số lượng mẫu: ", len(data))
# print("Số lượng thuộc tính: ", data.shape[1] - 1)  # Trừ đi 1 vì có cột 'Class'

# Tách dữ liệu thành X (các thuộc tính) và y (nhãn)
X = data.drop(columns=['Class'])
y = data['Class']

# Vẽ biểu đồ tần suất của biến mục tiêu
# plt.hist(y, bins=10)
# plt.title("Biểu đồ tần suất của biến mục tiêu")
# plt.xlabel("Giá trị")
# plt.ylabel("Tần suất")
# plt.show()

# Chia dữ liệu thành tập huấn luyện và kiểm tra (70% huấn luyện, 30% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo các mô hình
nb_model = GaussianNB()
logistic_model = LogisticRegression(max_iter=1000)
dt_model = tree.DecisionTreeClassifier(criterion="gini", max_depth=5,min_samples_leaf=15)
# knn_model = KNeighborsClassifier(n_neighbors=3)
# svm_model = svm.SVC(kernel='rbf')  #linear,rbf,poly
rfc_model = RandomForestClassifier(n_estimators=150, random_state=42)
bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(criterion="gini", max_depth=5,min_samples_leaf=15), n_estimators=150, random_state=42)
# boosting_model = AdaBoostClassifier(
#     estimator=DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=15),
#     n_estimators=150,
#     algorithm='SAMME',
#     random_state=42
# )


# # Khởi tạo bộ StandardScaler
scaler = MinMaxScaler()
# # Chuẩn hóa dữ liệu
X_train_std = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_std  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# # # Lưu scaler vào tệp
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

nb_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rfc_model.fit(X_train, y_train)
bagging_model.fit(X_train, y_train)
logistic_model.fit(X_train_std, y_train)
# # # # Lưu các mô hình vào file pickle
pickle.dump(nb_model, open('nb_model.pkl', 'wb'))
pickle.dump(dt_model, open('dt_model.pkl', 'wb'))

pickle.dump(rfc_model, open('rfc_model.pkl', 'wb'))
pickle.dump(bagging_model, open('bagging_model.pkl', 'wb'))
pickle.dump(logistic_model, open('logistic_model.pkl', 'wb'))

models = [nb_model,dt_model,rfc_model,bagging_model,logistic_model]
model_names = ['Naiv Bayes','Decision Tree','Random Forest','Bagging','LogisticRegression']
for i in range(len(models)):
    model = models[i]
    print('Mô hình: ', model_names[i])
    if (model == logistic_model):
        y_pred = model.predict(X_test_std)
    else:
        y_pred = model.predict(X_test)
    print("Độ chính xác của mô hình với phương pháp kiểm tra hold-out: %.3f" % accuracy_score(y_test, y_pred))
    nFold = 20
    scores = cross_val_score(model, X, y, cv=nFold)
    print("Độ chính xác của mô hình với nghi thức kiểm tra %d-fold: %.3f" % (nFold, np.mean(scores)))

