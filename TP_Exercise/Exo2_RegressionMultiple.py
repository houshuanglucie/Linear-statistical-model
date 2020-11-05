

"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Exercice 2 - Regression lineaire multiple et inference statistique (suite de l'exercice 1 / partie 3)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Dans le probleme suivant, on considere que l'on connait les notes moyennes sur l'année de n eleves dans p matieres, ainsi que leur note a un concours specifique en fin d'annee. On se demande si on ne pourrait pas prédire la note des etudiants au concours en fonction de leur moyenne annuelle afin d'estimer leurs chances au concours.

On va resoudre le probleme a l'aide de la regression lineaire multiple de scikitlearn (et plus a la main) et estimer a quel point les predictions sont precises par inference statistique.
"""

import numpy as np
import matplotlib.pyplot as plt


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Partie 1 -- Apprentissage/prediction :
# - Bien comprendre la fonction 'SimulateObservations2'
# - A l'aide de la fonction 'SimulateObservations2', simulez un jeu de donnees d'apprentissage [X_l,y_l] avec n_l=30 observations et un jeu de test [X_t,y_t] avec n_t=1000 observations. Les observations seront en dimension p=10.
# - Effectuez la regression lineaire multiple avec sklearn.linear_model.LinearRegression
# - representez un nuage de points dont chaque point a pour coordonnee (y_true,y_predicted). Les observations test seront utilisees. Calculez de meme la 'mean squared error' sur ces donnees  
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



def SimulateObservations2(n_train,n_test,p):
  """
  n_train: number of training obserations to simulate
  n_test: number of test obserations to simulate
  p: dimension of the observations to simulate
  """
  
  ObsX_train=20.*np.random.rand(n_train,p)
  ObsX_tst=20.*np.random.rand(n_test,p)
  
  RefTheta=np.random.rand(p)
  RefTheta=RefTheta/RefTheta.sum()
  print("The thetas with which the values were simulated is: "+str(RefTheta))
  
  ObsY_train=np.dot(ObsX_train,RefTheta.reshape(p,1))+1.5*np.random.randn(n_train,1)
  ObsY_tst=np.dot(ObsX_tst,RefTheta.reshape(p,1))+1.5*np.random.randn(n_test,1)
  
  return [ObsX_train,ObsY_train,ObsX_tst,ObsY_tst,RefTheta]

n=30
p=10

#TO DO
[X_l, y_l, X_t, y_t, RefTheta] = SimulateObservations2(n, 1000, p)


from sklearn.linear_model import LinearRegression

#TO DO
LR_regressor = LinearRegression()
LR_regressor.fit(X_l, y_l)

y_pred = LR_regressor.predict(X_t)

from sklearn.metrics import mean_squared_error

MSE = mean_square_error(y_t, y_pred)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Partie 2 -- Inference sur les erreurs : On fait l'hypotese que le bruit sur les observations est Gaussien (ce qui est vrai puisqu'on a simule les donnees comme ca). Nous allons alors etudier comment les erreurs d'approximation sont distribuees.
# - Utilisez les fonction np.histogram et plt.plot pour representer la distribution de l'erreur d'approximation dans les donnees de la partie 1.
# - La distribution de l'erreur est liee a une loi de student a n-p-1 degres de libertes. Nous n'allons pas caller cette loi a nos donnees mais simplement mesurer la moyenne de erreur au carre (MSE - ou biais) dans nos donnees pour evaluer a quel point notre methode est precise.

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# TO DO 

erreurs=[y_t - y_pred]

histo_erreurs = np.histogram(erreurs, bins=20)

MSE=#...






#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Partie 3 -- Variations de l'erreur pour differentes valeurs de n ou p : 
# - Reproduire les parties 1 et 2 de l'exercice pour different nombres d'observations d'apprentissage (n) et differentes dimensions des observations (p) :
#  [Tests 1] : (n=30,p=1) , (n=30,p=5) , (n=30,p=10) , (n=30,p=15) , (n=30,p=20) , (n=30,p=25) , (n=30,p=29)
#  [Tests 2] : (n=11,p=10) ,(n=15,p=10) ,(n=20,p=10) ,(n=30,p=10) , (n=60,p=10) , (n=100,p=10)  
#
# Aussi bien pour [Tests 1] que pour [Tests 2], verifiez comment evolue la MSE quand n ou p varie.
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



def GetMSE(n_train,n_test,p):
    [X_l, y_l, X_t, y_t, RefTheta] = SimulateObservations2(n_train,n_test, p)
    
    return MSE
  
  
#  [Tests 1]


MSE_30_1=GetMSE(30,1000,1)

#...TO DO


#  [Tests 2]

MSE_11_10=GetMSE(11,1000,10)

#...TO DO


