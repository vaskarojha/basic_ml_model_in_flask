import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
df = pd.read_csv('iris.csv')

# print(df.head(5))

# select dependent and independent variable as y and x respectively

X = df[["sepal_length" , "sepal_width" , "petal_length" , "petal_width"]]
y = df['variety']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=50)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#instantiate the model
classifier = RandomForestClassifier()

#fit the model
classifier.fit(X_train, y_train)


# make pickle file of the model
pickle.dump(classifier, open('model.pkl', 'wb'))
