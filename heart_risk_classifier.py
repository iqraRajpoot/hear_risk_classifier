import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Heart_disease_prediction:

    def read_data_heart(self):
        df = pd.read_csv('heart.csv')
        print("data", df.head())
        print("rows in the ds: ", df.shape[0])
        print("colms in the ds: ", df.shape[1])
        return df

    def check_null_values(self, df):
        print("data", df.head())
        null_values = df.isnull().any()
        return null_values
    
    def check_statistics(self, df):
        stats = df.describe().T
        return stats
    
    def data_visualization(self, df):
        fig = plt.figure(figsize= (15,15))
        ax = fig.gca()
        df.hist(ax=ax)
        plt.show() 
    
    def check_data_is_balance(self, df):
        graph = sns.countplot(x= 'target', data = df)
        plt.xlabel('target')
        plt.ylabel('count')
        plt.show()

    def check_duplicates(self, df):
        duplicates = df.duplicated()
        total_dupliates = df.duplicated().sum()
        if total_dupliates > 0:
             duplicate_records = df[df.duplicated()]
             print("----------------\n", duplicate_records)
        return duplicates, total_dupliates
    
    def drop_duplicates(self, df):
        df2 = df.drop_duplicates()
        return df2
    

class Feature_engineering(Heart_disease_prediction):
    def __init__(self,df2):
        super().__init__()
        self.df2 = df2

    def check_outliers(self):
        fig = plt.figure(figsize=(15,10))
        self.df2.boxplot(rot = 90)
        plt.title("outliers in the ds")
        plt.show()

    def check_correlation(self):
        fig = plt.figure(figsize=(10,10))
        sns.heatmap(self.df2.corr(), annot = True, cmap = 'coolwarm', fmt = '.2f')
        plt.title("Feature engineering heatmap")
        plt.show()

    def preprocessing (self):
        processed_df = pd.get_dummies(self.df2, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
        return processed_df

    def check_distribution(self, processed_df):
        processed_df.hist(figsize = (10,10), bins = 20)
        plt.show()

    def feature_scaling(self, processed_df):
        scaling = StandardScaler()
        scale_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        pre_processed[scale_columns] = scaling.fit_transform(pre_processed[scale_columns])
        return pre_processed.head()

    def model_building(self, processed_df):
        x = processed_df.drop(columns=['target'])
        y = processed_df['target']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)

        log_reg = LogisticRegression()
        svm = SVC()
        rfc = RandomForestClassifier(n_estimators=100, random_state= 42)
        knn = KNeighborsClassifier()

        models = {

            "logistic regression" : log_reg,
            "support vector machine" : svm,
            "random forest classifier" : rfc,
            "k nearest neighbor" : knn

        }
        accuracies = []
        for name, model in models.items():
            print(f"training {name} ...")
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

            print(f"{name} Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            print("="*50)
            

        plt.figure(figsize=(10,6))
        plt.bar(models.keys(), accuracies, color=['blue', 'green', 'red', 'orange'])

        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title("model comparisons on heart diseases ds")
        plt.ylim([0,1])
        plt.show()



obj = Heart_disease_prediction()
read_data = obj.read_data_heart()
null_val = obj.check_null_values(read_data)
print("------------------------------null value-----------------------------\n",null_val)
statistics = obj.check_statistics(read_data)
print("------------------------------statistics-----------------------------\n", statistics)
visualize = obj.data_visualization(read_data)
count_targets = obj.check_data_is_balance(read_data)
print(count_targets)
duplicated, total_dup = obj.check_duplicates(read_data)
print(duplicated,"-------", total_dup)
drop_duplicates = obj.drop_duplicates(read_data)
print("------------------------------drop duplicates-----------------------------\n", drop_duplicates)


#-------------class feature engineering----------------
obj2 = Feature_engineering(drop_duplicates)
outliers = obj2.check_outliers()
print(outliers)
print("------------------------------correlation-----------------------------\n")
corr = obj2.check_correlation()
print("------------------------------pre-processing-----------------------------\n")
pre_processed = obj2.preprocessing()
print("------------------------------distirbution-----------------------------\n")
distribution = obj2.check_distribution(pre_processed)
scaling = obj2.feature_scaling(pre_processed)
print("------------------------------scaling-----------------------------\n",scaling)
print("------------------------------model-------------------------------\n")
model = obj2.model_building(pre_processed)
print("------------------------------The End-------------------------------\n")


