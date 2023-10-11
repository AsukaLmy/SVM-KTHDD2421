import numpy as np
import random , math
from scipy.optimize import minimize
import matplotlib . pyplot as plt
import sys

#random spot generate
np.random.seed(233) #delete it when u need different spots
classA = np.concatenate(
    (np.random.randn(10,2)*0.3 + [2,0.5],
     np.random.randn(10,2)*0.3 + [-2,0.5]))
classB = np.random.randn(20,2)*0.3 + [0.0,-0.5]

inputs = np.concatenate((classA,classB))
targets = np.concatenate(
    (
        np.ones(classA.shape[0]),
        -np.ones(classB.shape[0])
    )
)

N = inputs.shape[0] #number of rows(sample)

permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute,:]
targets = targets[permute]





#kernel define
#Linear kernel
def LinearKernel(x, y):
    return np.dot(x, y)
    
# Polynomial kernels
def PolyKernel(x, y):
    #set p yourself, u know what will happen
    p = 2
    return np.power(np.dot(x, y) + 1, p)
    
def RBFKernel(x, y):
    #the parameter sigma is used to diside how beautiful the boundary is
    sigma = 2
    return math.exp(-math.pow(np.linalg.norm(np.subtract(x, y)), 2)/(2 * math.pow(sigma,2)))
#select your champion
Kernel = PolyKernel




#data input
#POWER!
Pmatrix = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        Pmatrix[i][j] = targets[i] * targets[j] * Kernel(inputs[i], inputs[j])

#used in minimize
def zerofun(alpha):
    return np.dot(alpha, targets)
    
def objective(alpha):
    return (1/2)*np.dot(alpha, np.dot(alpha, Pmatrix)) - np.sum(alpha)



#tricky 0
start = np.zeros(N)
C = 0.01

#upper constraint
#freedom always comes with constraint
#B = [(0, None) for b in range(N)]
B = [(0, C) for b in range(N)]

XC = {'type':'eq', 'fun':zerofun}

solutioncheck = 1

MiniMize = minimize(objective, start, bounds=B, constraints=XC)
if (not MiniMize['success']): 
    print('Cannot find optimizing solution')
    solutioncheck = 0
#    raise ValueError('Cannot find optimizing solution')


#failure pave the way to success
if (solutioncheck == 0):
    plt.plot(
        [p[0] for p in classA],
        [p[1] for p in classA],
        'b.'
    )
    plt.plot(
        [p[0] for p in classB],
        [p[1] for p in classB],
        'r.'
    )
    #plt.savefig('2.RBF_F1.png')
    plt.show()
    sys.exit()


#Extract non-zero alphas
#not everyone can find his match
alpha = MiniMize['x']
nonzero = [(alpha[i], inputs[i], targets[i]) for i in range(N) if abs(alpha[i]) > 0.00001]




#finally we can calculate w and b
def b_value():
    b_sum = 0
    for value in nonzero:
        b_sum += value[0] * value[2] * Kernel(value[1], nonzero[0][1])
    return b_sum - nonzero[0][2]
        
def indicator(x, y, b):
    totsum = 0
    for value in nonzero:
        totsum += value[0] * value[2] * Kernel([x, y], value[1])
    return totsum - b

#plot boundary
#plot spot
b = b_value()
plt.plot(
    [p[0] for p in classA],
    [p[1] for p in classA],
    'b.'
)
plt.plot(
    [p[0] for p in classB],
    [p[1] for p in classB],
    'r.'
)


xgrid = np.linspace(-5,5)
ygrid = np.linspace(-4,4)

grid = np.array(
    [
        [indicator(x,y,b) for x in xgrid] for y in ygrid
    ]
)

plt.contour(
    xgrid, ygrid, grid,
    (-1.0, 0.0, 1.0),
    colors=('red','black','blue'),
    linewidths = (1,3,1)
)
plt.axis('equal')
plt.savefig('4.Pk.p=2.C=0.01.png')
plt.show()