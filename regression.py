from sklearn.datasets import load_diabetes

import numpy as np 
import matplotlib.pyplot as plt 

diabetes = load_diabetes()

x = diabetes.data[:, np.newaxis, 2]
x = x.reshape(x.shape[0], )
y = diabetes.target

x_train = x[:100]
y_train = y[:100]

y_test = y[100:]
x_test = x[100:]


def rmse(y, y_hat):
	rmse = (1/len(y)) * sum([(y_ - y_model)**2 for y_ in y for y_model in y_hat])
	return rmse

def sgd(x, y):
	m = 0
	b = 0
	m_grad = 0
	b_grad = 0

	# nilai learning rate mempengaruhi hasil prediksi koefisien
	learning_rate = 0.001

	for i in range(len(x)):

		# hitung slope antara rmse terhadap m 
		m_grad =  (2/len(x)) * -x[i] * (y[i] - m*x[i]+b)

		# hitung slope antara rmse terhadap b
		b_grad =  (2/len(x)) * (-(y[i] * m*x[i]+b))
		
		# update coeffecients
		m += m - (learning_rate * m_grad)
		b += b - (learning_rate * b_grad)

		#print(m_grad, "\t", b_grad, '\t', b, '\t', m)
	
	return m, b


#print(sgd(x, y)) 
m, b = sgd(x_train, y_train)
y_model = m*x_train + b 

#print("rmse: ", rmse(y_train, y_model))
#print("m prediction: ",m,'\t', "b prediction :", b)

plt.plot(x_train, y_train, 'b*')
plt.plot(x_train, y_model, 'b-')

#plt.plot(x_train, y_model, 'r-')
plt.show()
