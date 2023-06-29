import numpy as np
import tensorflow as tf
from keras import Sequential,Input
from keras.layers import Dense
from keras.losses import MeanSquaredError,SparseCategoricalCrossentropy
from keras.activations import relu,linear
from keras.optimizers import Adam
from keras import regularizers
from sklearn.metrics import mean_squared_error

data=np.genfromtxt('Life Expectancy Data.csv',delimiter=',',skip_header=1)
print(data.shape)

x=np.delete(data,[0,1,2,17],axis=1)



empty_rows=np.isnan(x).any(axis=1)
print(empty_rows)
x=x[~empty_rows]
y=x[:,0]
y=np.reshape(y,(-1,1))
x=np.delete(x,[0],axis=1)
print(x.shape)

for i in range(x.shape[1]):
    average=np.average(x[:,i])
    std=np.std(x[:,i])
    #print(average,std)
    x[:,i]=(x[:,i]-average)/std

print(x)
v60=int(0.6*x.shape[0])
v80=int(0.8*x.shape[0])

X=x[:v60]
X_cv=x[v60:v80]
X_test=x[v80:]

print(X.shape)

Y=y[:v60]
Y_cv=y[v60:v80]
Y_test=y[v80:]

#print(x.shape,y.shape,x.dtype,y.dtype)
#if i increased the number of units to 10 in layer 1, it reduced the cost function even further but it did not make a significant difference than this one
#economic development, medical priority given, deaths, diseases, ill health
model=Sequential([
    Dense(units=5,activation=relu,kernel_regularizer=regularizers.l2(.1)),
    Dense(units=1,activation=linear,kernel_regularizer=regularizers.l2(.1))
])

#bad model used for trail
'''model=Sequential([
    Dense(units=1,activation=linear,kernel_regularizer=regularizers.l2(.1))
])'''

model.compile(
    loss=MeanSquaredError(),
    optimizer=Adam(0.001)
)

model.fit(X,Y,epochs=1000)

y_output_cv=model.predict(X_cv)
print(mean_squared_error(Y_cv,y_output_cv))
print(np.average(abs(Y_cv-y_output_cv)))

y_output_test=model.predict(X_test)
print(mean_squared_error(Y_test,y_output_test))
print(np.average(abs(Y_test-y_output_test)))

x_tp=np.array(X[1]).reshape((1,-1))

y_tp=np.array(Y[1]).reshape((1,-1))
y_output_tp=model.predict(x_tp)
print(y_tp,y_output_tp)
