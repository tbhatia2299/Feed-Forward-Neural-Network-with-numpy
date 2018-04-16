import numpy as np
import matplotlib.pyplot as plt

X=np.array((([.15,.4],[.14,.6],[.11,.7],[.10,.7],[.10,.9],[.9,.12],[.9,.14])))
Y=np.array(([.62],[.70],[.75],[.77],[.79],[.87],[.88]))

k=1
num_validation_samples=1
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

def relu(x):
    return x * (x > 0)

def derivatives_relu(x):
    return 1 * (x > 0)
    
    

for i in range(500000):

    #Forward Propagation
    r=np.dot(X,W1) #for hidden layer
    r1=r+B1
    r2=relu(r1)
    r3=np.dot(r2,W2)
    r4=r3+B2
    output=relu(r4)



    lr=0.001

    #Backpropagation
    E=Y-output
    slope_output_layer = derivatives_relu(output)
    delta_output = E * slope_output_layer
    Error_2=delta_output.dot(W2.T) #Contribution of hidden layer weights to the output error
    slope_hidden_layer=derivatives_relu(r2)
    d_hiddenlayer = Error_2 * slope_hidden_layer
    W1+=X.T.dot(d_hiddenlayer)*lr
    B1+= np.sum(delta_output, axis=0,keepdims=True) *lr
    W2+=r2.T.dot(delta_output)*lr

print('Output is', output)
plt.plot(E)
plt.show()
