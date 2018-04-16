import numpy as np
import matplotlib.pyplot as plt

X=np.array(([15,.4],[.14,.6],[.11,.7],[.10,.7],[.10,.9],[.9,.12],[.9,.14]), dtype=float)
Y=np.array(([[.62],[.70],[.75],[.77],[.79],[.87],[.88]]), dtype=float)

k=3
num_validation_samples=2
for fold in range(k):
   test_data=X[num_validation_samples*fold:num_validation_samples*(fold+1)]
   training_data=X[:num_validation_samples*fold]+X[num_validation_samples*(fold+1)]

   

inputSize=2
outputSize=1
hiddenSize=15


W1=np.random.randn(inputSize, hiddenSize)
B1=np.random.randn(1,hiddenSize)
W2=np.random.randn(hiddenSize, outputSize)
B2=np.random.randn(1, outputSize)

def sigmoid (x):
   return 1/(1 + np.exp(-x))

def derivatives_sigmoid(x):
      return x * (1 - x)

for i in range(500000):

    #Forward Propagation
   r=np.dot(X,W1) #for hidden layer
   r1=r+B1
   r2=sigmoid(r1)
   r3=np.dot(r2,W2)
   r4=r3+B2
   output=sigmoid(r4)
   
    

   lr=0.01
   
   #Backpropagation
   E=Y-output
   slope_output_layer = derivatives_sigmoid(output)
   delta_output = E * slope_output_layer
   Error_2=delta_output.dot(W2.T) #Contribution of hidden layer weights to the output error
   slope_hidden_layer=derivatives_sigmoid(r2)
   d_hiddenlayer = Error_2 * slope_hidden_layer
   W1+=X.T.dot(d_hiddenlayer)*lr
   B1+= np.sum(delta_output, axis=0,keepdims=True) *lr
   W2+=r2.T.dot(delta_output)*lr
   
   
print('Output is', output)
plt.plot(E)
plt.show()


X=np.array(([.15,.4],[.14,.6],[.11,.7],[.10,.7],[.10,.9],[.9,.12],[.9,.14]), dtype=float)
Y=np.array(([.62],[.70],[.75],[.77],[.79],[.87],[.88]), dtype=float)

k=3
num_validation_samples=2
for fold in range(k):
   test_data=X[num_validation_samples*fold:num_validation_samples*(fold+1)]
   training_data=X[:num_validation_samples*fold]+X[num_validation_samples*(fold+1)]

inputSize=2
outputSize=1
hiddenSize=15
        
W1=np.random.randn(inputSize, hiddenSize)
B1=np.random.randn(1,hiddenSize)
W2=np.random.randn(hiddenSize, outputSize)
B2=np.random.randn(1, outputSize)

def tanh(x):
    return np.tanh(x)

def derivatives_tanh(x):
    return 1.0 - np.tanh(x)**2




for i in range(50000):

    #Forward Propagation
    r=np.dot(X,W1) #for hidden layer
    r1=r+B1
    r2=tanh(r1)
    r3=np.dot(r2,W2)
    r4=r3+B2
    output1=tanh(r4)



    lr=0.01

    #Backpropagation
    E1=Y-output1
    slope_output_layer = derivatives_tanh(output)
    delta_output = E * slope_output_layer
    Error_2=delta_output.dot(W2.T) #Contribution of hidden layer weights to the output error
    slope_hidden_layer=derivatives_tanh(r2)
    d_hiddenlayer = Error_2 * slope_hidden_layer
    W1+=X.T.dot(d_hiddenlayer)*lr
    B1+= np.sum(delta_output, axis=0,keepdims=True) *lr
    W2+=r2.T.dot(delta_output)*lr

print('Output is', output1)
print('Error is', E1)

plt.plot(E1)
plt.show()



        
