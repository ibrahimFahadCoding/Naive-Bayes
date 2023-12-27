import pandas
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = pandas.read_csv('iris.csv')
x = data.drop('species', axis=1).values
y = data['species'].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=66)

datapoint = [5.3, 9, 1.2, 7.4]

model = MultinomialNB()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

print(f"Accuracy Score: {accuracy_score(y_test, predictions)*100}%")

datapoint_pred = model.predict([datapoint])
print(datapoint_pred)

