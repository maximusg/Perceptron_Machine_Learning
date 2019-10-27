import numpy as np
x=[.1,.2,.4,.5,.34,.56,.46,0] #last input is y actual
w1 =[.3,.1,.2,.3,.4,.5,.6,.7] #first col is bias
w2 =[.4,.123,.354,.34,.3,.35,.23,.345]
w3 =[.10,.45,.4,.2,.3,.4,.324,.23]


def feed_fwd(): #takes an input gen an output
    pass
def backprop(): #trains model by adj weights
    #use stochastic gradient descent to optimzie weights

    pass

#rectified linear units function ReLU 
def predict(row, weights):
    # row = expandS(row,2)
    activation = weights[0]
    # print("row length ",len(row))
    # print("weights length", len(weights))
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

print("xinput ",x[-2:])
# print("w1input ",w1[1:])
# # z1=predict(x,w1) #z1 is first guess
# z1 = np.dot(x[:-1:],w1[1:])+w1[0]
# print("z1 ", z1)
# a1 = np.tanh(z1)
# print("a1",a1) #first activation
# z2 = np.dot([a1],w2[-1:])+w2[0]
# print("z2", z2)
# a2 = np.tanh(z2)
# print("a2",a2) #first activation
# z3 = np.dot([a2],w3[-1:])+w3[0]
# print("z3", z3)

# def expand(x):
#     ary=[]
#     for i in range(0,len(x)):
#         for j in range(0, len(x)):
#             new = x[i]*x[j]
#             ary.append(new)
    
#     return ary
# def expandS(x,s):
#     for i in range(0,s):
#         x=expand(x)
#     return x
    
# print(expandS(x,2))

# print(type(1)==int)