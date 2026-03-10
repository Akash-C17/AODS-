import numpy as np 
import matplotlib.pyplot as plt

X= np.array([1,2,3,4,5,])
y= np.array([2,4,6,8,10])

m=0
b=0
learning_rate=0.01
epochs= 1000
n = len(X)

for _ in range(epochs):
    y_pred = m*X+b
    error = y_pred - y

dm = (2/n)*np.sum(X*error)
db = (2/n)*np.sum(error)
    

m-= learning_rate*dm
b-= learning_rate*db 

print(f'Optimized Slope(m:{m:.4f})')
print(f'Optimized Intercept(b): {b:.4f}')

plt.scatter(X,y, color="red", label= "Actual Data")
plt.plot(X,m*X + b , color= 'blue', label="Best-Fit Line")
plt.xlabel('y')
plt.legend()
plt.show()