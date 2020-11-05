
"""
EXERCICE 6 : PLS

inspire de http://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_compare_cross_decomposition.html
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# regression PLS avec reponse univariee (PLS1)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#Generation de donnees
n = 1000
p = 10
X = np.random.normal(size=n * p).reshape((n, p))
y = X[:, 0] + 2 * X[:, 1] + 0.5 * X[:, 2]*X[:, 3] + np.random.normal(size=n) + 5

#QUESTION 1: Comment sont construites les donnees simulees ? 
#            Que vous attendez-vous a voir dans la PLS



#QUESTION 2: Quelle est la signification de pls1.x_rotations_. Comment l'interpretez vous ?
#            Que renvoi de plus 'pls1.predict(X)' ? Comparez ce resultat 
#            a y.


#QUESTION 3: Est-ce qu'une regression lineaire multiple avec selection de modele
#            conduirait a des resultats similaires ?


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# regression PLS avec reponse multivariee (PLS2)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

n = 1000
q = 3
p = 10
X = np.random.normal(size=n * p).reshape((n, p))
B = np.array([[1., 2.] + [0] * (p - 2)] * q)
B[1,0]/=2.
B[2,1]/=2.
Y = np.dot(X, B.transpose()) + np.random.normal(size=n * q).reshape((n, q))/2

#QUESTION 4: Comment sont construites les donnees simulees ? 


#QUESTION 5: predisez maintenant Y a l'aide de X. Qu'observez vous.
