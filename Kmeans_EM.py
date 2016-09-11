#---------------------------------
# K-means clustering
# Author: Kaushik Balakrishnan
#---------------------------------

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pandas import DataFrame
from numbers import Integral
from matplotlib.pyplot import cm 
from numpy.linalg import inv, det


def plotSol(K,X,mu,z,pause):
  plt.figure(2)
  color=color = ["red", "green", "blue", "orange", "yellow", "violet", "brown"] #iter(cm.rainbow(np.linspace(0,1,10)))
  plt.ion() # for interactive plotting in real time

  plt.subplot(211) 
  for k in range(K):
    x = X[z == k]
    print("number of points in cluster {} (Kmeans) = {}".format(k,x.shape[0]))
    plt.plot(x[:,0], x[:,1],marker='o',c=color[k],linestyle='None',markersize=5)
  plt.plot(mu[:,0], mu[:,1],'ko',markersize=12)
  plt.title('Kmeans')

  print('-----------')

  plt.subplot(212) 
  for k in range(K):
    x = X[z == k]
    print("number of points in cluster {} (EM) = {}".format(k,x.shape[0]))
    plt.plot(x[:,0], x[:,1],marker='o',c=color[k],linestyle='None',markersize=5)
  plt.plot(mu[:,0], mu[:,1],'ko',markersize=12)
  plt.title('EM')


   
  plt.show()
  if (pause == 0):
   plt.pause(0.05)
  else: 
   plt.pause(10) 
  plt.clf() 

#-------------------------------------------------------------

def Kmeans(D, N, K, X, mu, z, random_state=0):
  
    if random_state is None:
        random_state = np.random.mtrand._rand
    elif isinstance(random_state, Integral):
        random_state = np.random.RandomState(random_state)


    # distance from centroid
    dist = np.zeros(K)
   
    # running sum
    sumX = np.zeros((K,D))

    # number of data points in each cluster
    Nk = np.zeros(K)

# iterate over all N samples  

    for i in range(N):     

     # Euclidean distance         
     for k in range(K):
       dist[k] = np.sum((X[i,:] - mu[k,:])**2.0)
  
     # find cluster for this data point 
     z[i] = np.argmin(dist)
     Nk[z[i]] += 1
     sumX[z[i],:] += X[i,:]
     
     
# re-compute mu 
    for k in range(K):       
       mu[k,:] = sumX[k,:]/Nk[k]
   

    return mu, z
 
#-------------------------------------------------------------

def compute_p(D,x,mu,sigma):     
     fact = 1.0/np.sqrt((2.0*np.pi)**D)    

     S = np.zeros((D,D))
     for i in range(D):
      S[i,i] = sigma[i]
     k = D 
     for i in range(D):
      for j in range(i+1,D):
       S[i,j] = sigma[k]
       S[j,i] = S[i,j]
       k += 1
     S = np.matrix(S)

     x_minus_mu = x[:] - mu[:]
     x_minus_mu = np.matrix(x_minus_mu)     

     if(det(S) <= 0.0):
       print('error in det(S) ',det(S))
       print('S=',S)
       exit()

     fact1 = 1.0/np.sqrt(det(S))

     return fact*fact1*np.exp(np.asscalar(-0.5*x_minus_mu*inv(S)*np.transpose(x_minus_mu)))
      


def getCluster(p):
    K, = p.shape
    minp = 0.0
    r = np.random.uniform()
    cluster = -1 
    for i in range(K):
      maxp = minp + p[i]
      if (r>=minp and r < maxp):
        cluster = i
      minp = maxp
    if(cluster < 0 or cluster > 2):
          print("error ",cluster,r,p) 
          exit()
    return cluster
    


def EM(D, N, K, X, mu, sigma, pik, z, random_state=0):   

    if random_state is None:
        random_state = np.random.mtrand._rand
    elif isinstance(random_state, Integral):
        random_state = np.random.RandomState(random_state)


    # number of data points in each cluster
    Nk = np.zeros(K)

    # Gaussian probability
    p = np.zeros(K)

    # summuk
    summuk = np.zeros((K,D)) 

    # sumsigma 
    sumsigma = np.zeros((K,3*(D-1)))


# iterate over all N samples  

    for i in range(N): 
     for k in range(K): 
       p[k] = pik[k]*compute_p(D,X[i,:],mu[k,:],sigma[k,:])
     p = p/np.sum(p)   
    
     z[i] = getCluster(p)
     Nk[z[i]] += 1.0
     summuk[z[i],:] += X[i,:]
     for j in range(D):
      sumsigma[z[i],j] += (X[i,j]-mu[z[i],j])**2.0 
     sumsigma[z[i],j+1] += (X[i,0]-mu[z[i],0])*(X[i,1]-mu[z[i],1])    

    for k in range(K):
     for j in range(D):
      mu[k,j] = summuk[k,j]/Nk[k]
     for j in range(3*(D-1)): 
      sigma[k,j] = sumsigma[k,j]/Nk[k]
     pik[k] = Nk[k]/np.sum(Nk[:])    
 

    return mu, sigma, z, pik

#-------------------------------------------------------------

X = pd.read_table("X.tsv", sep="\t",header=None)
X = np.array(X)
print('X shape = ',X.shape)

N,D = X.shape

print('N=',N)
print('D=',D)

# `K` is the number of clusters
K = 3

print('K=',K)

#----------------------------------------------------------

# initial conditions

random_state = 0

if random_state is None:
  random_state = np.random.mtrand._rand
elif isinstance(random_state, Integral):
  random_state = np.random.RandomState(random_state)


# initial mu
mu1 = random_state.multivariate_normal(np.zeros(D), np.eye(D), size=K)
mu2 = random_state.multivariate_normal(np.zeros(D), np.eye(D), size=K)

# cluster label 
z1 = np.zeros(N)
z2 = np.zeros(N)

# sigma
sigma = np.zeros([K,3*(D-1)])
sigma[:,:D] = np.abs(np.random.randn(K,D))
# set sigma > 0 initially so that matrix is positive definite

# initial pik (Gaussian fractions)
pik = np.ones(K)/K

#-------------------------------------------------------

nepochs = 25

for epoch in range(nepochs):

 print('----------------------------------------') 
 print('epoch # = ', epoch)

 
 # call Kmeans()
 mu1, z1 = Kmeans(D, N, K, X, mu1, z1, random_state=0)

 print('--------------------------')
 print('mu (Kmeans) =', mu1)

 # call EM()

 # early iterations use K-means mu 
 if(epoch < 3):
   mu2 = mu1
   sigma[:,2] = 0.0 
 
 mu2, sigma, z2, pik = EM(D, N, K, X, mu2, sigma, pik, z2, random_state=0)

 print('--------------------------')
 print('mu (EM) =', mu2)
 print('sigma (EM)', sigma)  
 print('pik (EM) =', pik)

 print('--------------------------')
 # visualize data
 pause = 0  
 if (epoch == nepochs-1):
   pause = 1 
 plotSol(K,X,mu1,z1,pause)

print('--------------------------')
#-------------------------------------------------------
