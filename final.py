import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense

df_train = pd.read_csv('train.csv')
X = df_train.iloc[:, 1:15].values
y = df_train.iloc[:, 15].values

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])
labelencoder_X_7 = LabelEncoder()
X[:, 7] = labelencoder_X_7.fit_transform(X[:, 7])

X_train = X
y_train = y

sc = StandardScaler()
X_train = sc.fit_transform(X_train)

classifier = Sequential()
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 14))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy')
classifier.fit(X_train, y_train, batch_size = 2, nb_epoch = 100)
classifier.save('my_model.h5')
#y_pred = classifier.predict(X_test)
#print(y_pred)
#final = pd.read_csv("test.csv")
#final['Evaded Tax'] = y_pred
#final.set_index('RowNumber', inplace=True)
#final.sort_values('Evaded Tax',ascending = False , inplace = True)
#final.to_csv('final_ans.csv',index = False)  # output file