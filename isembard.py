import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson,skellam
import scipy.optimize as spo
from alive_progress import alive_bar
from time import sleep
import itertools
from itertools import product
 
np.set_printoptions(suppress=True)
max_goals=10
 
def alphamdl(LAM):
   home_goals=np.vstack(range(max_goals+1))
   away_goals=range(max_goals+1)
 
   return np.multiply(poisson.pmf(home_goals,LAM[0]),poisson.pmf(away_goals,LAM[1]))
 
def betamdl(LAM):
 
   def G(K):
       m=np.zeros(np.array(K)+[1,1])
       for I in np.array(list(product(range(K[0]+1),range(K[1]+1))),dtype=int):
           m[I[0]][I[1]]=max(min(poisson.cdf(K[0]-I[0],LAM[2]),1-poisson.cdf(K[1]-I[1]-1,LAM[2]))-max(poisson.cdf(K[0]-I[0]-1,LAM[2]),1-poisson.cdf(K[1]-I[1],LAM[2])),0)
       return m
 
   m=np.zeros((max_goals+1,max_goals+1))
 
   for I in np.array(list(product(range(max_goals+1),repeat=2)),dtype=int):
       home_goals=np.vstack(range(I[0]+1))
       away_goals=range(I[1]+1)
 
       m[I[0]][I[1]]=np.sum(np.multiply(np.multiply(poisson.pmf(home_goals,LAM[0]),poisson.pmf(away_goals,LAM[1])),G(I)))   
   return m 
 
def karlismdl(LAM):
 
   def G(K):
      
       I=range(min(K)+1)
       return np.dot(np.multiply(poisson.pmf(K[0]-I,LAM[0]),poisson.pmf(K[1]-I,LAM[1])),poisson.pmf(I,LAM[2]))
 
   m=np.zeros((max_goals+1,max_goals+1))
 
   for I in np.array(list(product(range(max_goals+1),repeat=2)),dtype=int):
       home_goals=np.vstack(range(I[0]+1))
       away_goals=range(I[1]+1)
 
       m[I[0]][I[1]]=G(I)
   return m 
 
def home_x_away(M):
   return [np.sum(np.tril(M,-1)),np.sum(np.diag(M)),np.sum(np.triu(M,1))]
 
def goals(M,g):
   return [np.sum(np.tril(np.fliplr(M),np.size(M,0)-g-2)),np.sum(np.triu(np.fliplr(M),np.size(M,0)-g-1))]
 
def BTTS(M):
   return [1-np.sum(M[0])-np.sum(M[:,0])+M[0][0],np.sum(M[0])+np.sum(M[:,0])-M[0][0]]
 
def nilnil(M):
   return M[0][0]
 
def expgoals(M):
   tot_goals=np.multiply(np.vstack(np.ones(max_goals+1)),range(max_goals+1))+np.transpose(np.multiply(np.vstack(np.ones(max_goals+1)),range(max_goals+1)))
   return np.sum(np.multiply(tot_goals,M))
 
def modcov(M,LAM):
    c=np.multiply(np.multiply(np.vstack(range(max_goals+1)),range(max_goals+1)),M)
    return np.sum(c)-(LAM[0]+LAM[2])*(LAM[1]+LAM[2])
 
def callmdl(model,LAM,matrix=np.empty(0),ss=False):
   
   if np.size(matrix)==0:
      matrix=model(LAM)

   dutch=np.round(np.reciprocal(matrix),2)
  
   res = {
   "Lambdas" : np.round(LAM,2),
   "Home-Draw-Away Odds" : np.round(np.reciprocal(home_x_away(matrix)),2),
   "Over/Under 1.5 Goals" : np.round(np.reciprocal(goals(matrix,1)),2),
   "Over/Under 2.5 Goals" : np.round(np.reciprocal(goals(matrix,2)),2),
   "Over/Under 3.5 Goals" : np.round(np.reciprocal(goals(matrix,3)),2),
   "BTTS Yes/No" : np.round(np.reciprocal(BTTS(matrix)),2),
   "Away Clean Sheet" : round(1/(np.sum(matrix[0,:])),2),
   "Expected Goals" : np.round(expgoals(matrix),2),
   "H-A Covariance" : np.round(modcov(matrix,LAM),2)
   }
 
   print("-" * 50)
   print("Results:")
   print("-" * 50)
   for n,r in res.items():
       print(n,"=",r)
   print("-" * 50)
   if ss==True:
       print("Spreadsheet Odds:")
       print("-" * 50)
       print(dutch[0][0])
       print(dutch[0][1])
       print(dutch[0][2])
       print(dutch[0][3])
       print(dutch[1][0])
       print(dutch[1][1])
       print(dutch[1][2])
       print(dutch[1][3])
       print(dutch[2][0])
       print(dutch[2][1])
       print(dutch[2][2])
       print(dutch[2][3])
       print(dutch[3][0])
       print(dutch[3][1])
       print(dutch[3][2])
       print(dutch[3][3])
       print("-" * 50)
 
def iSeMBard(TuneVars,method='cov',showres=False,model=None):
   
   if model is None:
      if TuneVars[2]<0:
         model=betamdl
      else:
         model=karlismdl

   if method=='cov':
      G=abs(np.array([TuneVars[2],TuneVars[2],TuneVars[2]]))**-0.5
   else:
      G=[0.5,0.5,0.5]

   def optimiser(LAM):
      matrix=model(LAM)

      if method=='cov':
         return [home_x_away(matrix)[0],home_x_away(matrix)[2],modcov(matrix,LAM)]-np.reciprocal(TuneVars)
      else:
         return [home_x_away(matrix)[0],home_x_away(matrix)[2],goals(matrix,2)[0]]-np.reciprocal(TuneVars)
 
   output=spo.root(optimiser,G)

   lambdas=output.x

   matrix=model(lambdas)

   if showres==True:
      print(output.message)
      callmdl(model,lambdas,matrix)

   if method=='cov':
      return 1/(goals(matrix,2)[0])
   else:
      return modcov(matrix,lambdas)

model=betamdl
 
h_odds=2.18
a_odds=4.25
d_odds=1/(1-(1/h_odds+1/a_odds))
over25=1.91
cov=-0.1

#iSeMBard([h_odds,a_odds,1/cov],showres=True)