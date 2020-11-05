
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score

"""
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PARTIE 1 : Utilisation de scikit-learn pour la regression lineaire
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

#generation de donnees test
n = 100
x = np.arange(n)
y = np.random.randn(n)*30 + 50. * np.log(1 + np.arange(n))

# instanciation de sklearn.linear_model.LinearRegression
lr = LinearRegression()   # create a model 
lr.fit(x[:, np.newaxis], y)  # np.newaxis est utilise car x doit etre une matrice 2d avec 'LinearRegression'
                    # lr.fit用来训练模型，np.newaxis 用来增加一个维度

# representation du resultat
# fig = plt.figure()
# plt.plot(x, y, 'r.')
# plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
# plt.legend(('Data', 'Linear Fit'), loc='lower right')
# plt.title('Linear regression')
# plt.show()


#QUESTION 1.1 : 
#Bien comprendre le fonctionnement de lr, en particulier lr.fit et lr.predict
print('-----------------------------------------------------')
print('Q1.1: lr.fit for training model，lr.predict for predicting data.')


#QUESTION 1.2 :
#On s'interesse a x=105. En supposant que le model lineaire soit toujours 
#valide pour ce x, quelle valeur correspondante de y vous semble la plus 
#vraisemblable ? 

print('Q1.2: La valeur correspondante de y avec x=105 est: y = %.10f' % lr.predict([[105]]))


"""
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PARTIE 2 : impact et detection d'outliers
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

#generation de donnees test
n = 10
x = np.arange(n)
y = 10. + 4.*x + np.random.randn(n)*3. 
y[9]=y[9]+20   ###【异常数值】

# instanciation de sklearn.linear_model.LinearRegression
lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)  # np.newaxis est utilise car x doit etre une matrice 2d avec 'LinearRegression'


# representation du resultat

# lr中的调用方法：
#   - lr.coef_：训练后的输入端模型系数，如果label有两个，即y值有两列。那么是一个2D的array
#   - lr.intercept_：截距(b_0)
print('-----------------------------------------------------')
b_0 = lr.intercept_
b_1 = lr.coef_[0]
print('b_0 = ' + str(b_0) + ', b_1= ' + str(b_1))

# fig = plt.figure()
# plt.plot(x, y, 'r.')
# plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
# plt.legend(('Data', 'Linear Fit'), loc='lower right')
# plt.title('Linear regression')
# plt.show()


#QUESTION 2.1 : 
#La ligne 'y[9]=y[9]+20' genere artificiellement une donnee aberrante.
#-> Tester l'impact de la donnee aberrante en estimant b_0, b_1 et s^2 
#   sur 5 jeux de donnees qui la contiennent cette donnee et 5 autres qui
#   ne la contiennent pas (simplement ne pas executer la ligne y[9]=y[9]+20).
#   On remarque que $\beta_0 = 10$, $\beta_1 = 4$ et $sigma=3$ dans les 
#   données simulees.

# 打印均方误差
print('-----------------------------------------------------')
ss = mean_squared_error(y,lr.predict(x.reshape(n,1)))   # Mean squared error
print('Mean squared error: %.8f' % ss)  ##((y-lr.predict(x))**2).mean()
sigma = np.sqrt(ss)
print('sigma: %.8f' % sigma) 


#QUESTION 2.2 : 
#2.2.a -> Pour chaque variable i, calculez les profils des résidus 
#         $e_{(i)j}=y_j - \hat{y_{(i)j}}$ pour tous les j, ou   
#         \hat{y_{(i)j}} est l'estimation de y_j a partir d'un modele  
#         lineaire appris sans l'observation i.
#2.2.b -> En quoi le profil des e_{(i)j} est different pour i=9 que pour  
#         les autre i
#2.2.c -> Etendre ces calculs pour définir la distance de Cook de chaque 
#         variable i
#
#AIDE : pour enlever un element 'i' de 'x' ou 'y', utiliser 
#       x_del_i=np.delete(x,i) et y_del_i=np.delete(y,i) 

print('-----------------------------------------------------')
for i in range(n):
    print('>>> For deleting i = ' + str(i) + ', I only print the ninth row of the data e_ij for comparison: ')
    x_del_i = np.delete(x,i)
    y_del_i = np.delete(y,i) 
    y_hat = b_0 + b_1 * x_del_i.reshape(n-1,1)
#     print(y_hat)
    e_ij = y_del_i - y_hat
    print(e_ij[-1:])
print('结论：删除第九个数据 和 删除其他数据 所得到的结果差别很大，可以说明第九个数据是个【奇异值】。')

print('-----------------------------------------------------')
for i in range(n):
    x_del_i = np.delete(x,i)
    y_del_i = np.delete(y,i) 
    y_hat = b_0 + b_1 * x_del_i.reshape(n-1,1)
    D_ij = np.sum((y_del_i - y_hat)**2) / (2 * ss)
    print('>>> For deleting i = ' + str(i) + ', Distance de Cook = ' + str(D_ij))


"""
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PARTIE 3 : Vers la regression lineaire multiple et optimisation
#
#On considere que l'on connait les notes moyennes sur l'annee de n eleves 
#dans p matieres, ainsi que leur note a un concours en fin d'annee. On 
#se demande si on ne pourrait pas predire la note des etudiants au 
#concours en fonction de leur moyenne annuelle afin d'estimer leurs 
#chances au concours.
#
#On va resoudre le probleme a l'aide de la regression lineaire en 
#dimension p>1 sans utiliser scikit-learn. 
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

#Question 1 :
# - A l'aide de la fonction 'SimulateObservations', simulez un jeu de donnees d'apprentissage [X_l,y_l] avec 30 observations et un jeu de test [X_t,y_t] avec 10 observations. Les observations seront en dimension p=10


def SimulateObservations(n_train,n_test,p):
  """
  n_train: number of training obserations to simulate
  n_test: number of test obserations to simulate
  p: dimension of the observations to simulate
  """
  
  ObsX_train=20.*np.random.rand(n_train,p)
  ObsX_tst=20.*np.random.rand(n_test,p)
  
  RefTheta=np.random.rand(p)**3
  RefTheta=RefTheta/RefTheta.sum()
  print("The thetas with which the values were simulated is: "+str(RefTheta))
  
  ObsY_train=np.dot(ObsX_train,RefTheta.reshape(p,1))+1.5*np.random.randn(n_train,1)
  ObsY_tst=np.dot(ObsX_tst,RefTheta.reshape(p,1))+1.5*np.random.randn(n_test,1)
  
  return [ObsX_train,ObsY_train,ObsX_tst,ObsY_tst,RefTheta]




#Question 2 :
# - On considere un modele lineaire en dimension p>1 mettre en lien les x[i,:] et les y[i], c'est a dire que np.dot(x[i,:],theta_optimal) doit etre le plus proche possible de y[i] sur l'ensemble des observations i. Dans le modele lineaire multiple, theta_optimal est un vecteur de taille [p,1] qui pondere les differentes variables observees (ici les moyennes dans une matiere). Coder alors une fonction qui calcule la moyenne des differences au carre entre ces valeurs en fonction de theta.

def CptMSE(X,y_true,theta_test):
  #TO DO
  
  return MSE



#Question 3 -- option 1 :
# - On va maintenant chercher le theta_test qui minimise cette fonction (il correspondra a theta_optimal), et ainsi résoudre le probleme d'apprentissage de regression lineaire multiple. Utiliser pour cela la fonction minimize de scipy.optimize


from scipy.optimize import minimize


#TO DO


#Question 3 -- option 2 :
#De maniere alternative, le probleme peut etre resolu a l'aide d'une methode de descente de gradient codee a la main, dans laquelle les gradients seront calcules par differences finies.




"""
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PARTIE 4 : maximum de vraisemblance
# - Tirer 10 fois une piece a pile ou face et modeliser les resultats obtenus comme ceux
#d'une variable aleatoire X qui vaut X_i=0 si on a pile et X_i=1 si on a face.
# - Calculer le maximum de vraisemblance du parametre p d'un loi de Bernoulli qui modeliserait le probleme.
# - Vérifier empiriquement comment évolue ce maximum de vraisemblance si l'on effectue de plus en plus de tirages
# - Que se passe-t-il quand il y a trop de tirages ? Représenter la log-vraisemblance plutot que la vraisemblance dans ce cas.
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


