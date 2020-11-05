#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#EXAMEN 2020 : Cet examen est composé de deux exercices. Les réponses seront données dans 
#un notebook et l'examen sera fait soit seul soit en binome. 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Exercice 1 : Nous souhaitons maintenant evaluer si un nouveau traitement a un effet significatif  
# sur l'efficacite d'un moteur en fonction de son age.  
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#QUESTION 1.1 : Lisez le fichier Observations_2.csv qui contient les donnees, traitez les et
#representez les avec :
# -> 'Age' en abscisse
# -> 'Efficiency' en ordonnee
# -> Les observations avec le traitement standard (Standard) en bleu
# -> Les observations avec le nouveau traitement (Tested) en rouge.
#A la vue du graphe, vous semble-t-il y avoir un effet ?



dataframe=pandas.read_csv("./Observations_2.csv",sep=' ')



#QUESTION 1.2 : On supposera qu'il existe une relation lineaire entre l'age du moteur et son
#niveau d'efficacite a un bruit Gaussien pres. Utilisez un modele de type regression 
#lineaire pour mettre en lien les données. Utilisez ensuite un test statistique pour 
#evaluer si l'impact du traitement est significatif.



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Exercice 2 : Nous souhaitons développer une méthode de type apprentissage automatique
#qui quantifie en temps reel le risque de collision d'un drone avec des objets environnants.
# 
#Pour y arriver, nous avons embarqué 18 capteurs sur un drone en phase de test. Un expert
#a alors quantifié à plusieurs instants son risque de collision avec un autre objet.  
#Un total de 67 observations labellisées ont été enregistrées dans le fichier 
#'Observations_1.csv'.
#
#Nous allons évaluer dans cet exercice si ces observations nous permettent de mettre en
#lien les données capteurs avec le niveau de risque. Nous allons aussi évaluer si nous 
#aurions une bonne performance avec un sous ensemble pertinent des capteurs. Chaque
#capteur a en effet un poids non négligeable et à besoin d'énergie pour fonctionner, ce qui
#a un impact sur le temps d'utilisation du drone entre deux rechargements de ses batteries.  
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

dataframe=pandas.read_csv("./Observations_1.csv",sep=';')

listColNames=list(dataframe.columns)


XY=dataframe.values
ColNb_Y=listColNames.index('Risk_level')


Y=XY[:,ColNb_Y].reshape((XY.shape[0],1))   #reshape is to make sure that Y is a column vector
X = np.delete(XY, ColNb_Y, 1)

listColNames.pop(ColNb_Y)     #to make it contains the column names of X only


for Col in range(len(listColNames)):
  plt.plot(X[:,Col],Y[:],'.')
  plt.xlabel(listColNames[Col])
  plt.ylabel('Risk_level')
  plt.show()



#QUESTION 2.1 : Observez les donnees unes par unes. Est-ce que vous identifiez visuellement des liens entre 
#certaines variables et la variable 'Risk_level'. Si oui, lesquels ?


#QUESTION 2.2 :   On se demande si il est possible de predire le niveau de 'Risk_level' à partir d'une
#               seule des variables 'Feature_01', 'Feature_07' ou 'Feature_16'. 
#
#QUESTION 2.2.1 : Effectuez une regression lineaire simple entre 'Risk_level' et chacune de ces 
#               variables.  Toutes les donnees seront utilisees. Evaluez alors la qualité des predictions a
#               sur toutes les donnees l'aide de la moyenne de l'erreur de prediction au carre (MSE). Quel
#               est le risque potentiel en utilisant cette stratégie de validation de l'apprentissage ? 
#
#QUESTION 2.2.2 : Evaluez a quel point les predictions sont stables a l'aide d'une methode de validation croisee
#               de type 4-folds.
#
#QUESTION 2.2.3 : Peut-on enfin dire si on observe une relation significative entre 'Risk_level'
#               et (independament) 'Feature_01', 'Feature_07' ou bien 'Feature_16'. On peut le valider
#               a l'aide d'un test d'hypothese dont on decrira la procedure.



#QUESTION 2.3 :   On s'interesse maintenant au lien entre la variable 'Risk_level' et 'Feature_12'.
#               On peut remarquer que ces donnees contiennent deux valeurs aberrantes.
#
#QUESTION 2.3.1 : Definissez une procedure pour detecter automatiquement deux donnees aberrantes dans  
#               un jeu de donnees. 
#
#QUESTION 2.3.2 : Nous supprimerons dans la suite de cet exercice les deux observations qui sont aberrantes sur 
#               la variable 'Feature_12'. Comment auriez-vous traite ces observations si vous aviez absolument
#                voulu preserver l'information qu'elles contiennent dans les autres variables ?


#QUESTION 2.4 :   Une fois les deux observations aberrantes de 'Feature_12' supprimees, on souhaite selectionner les
#               variables de 'X' qui permettent de prédire au mieux 'Risk_level' a l'aide de la 
#               regression multiple regularisee.

#QUESTION 2.4.1 : Quelle strategie vous semble la plus appropriee pour selectionner les variables les plus 
#               pertinentes ? Quel pretraitement allez-vous de meme effectuer sur les donnees.

#QUESTION 2.4.2 : Effectuez la procedure de selection des variables optimales en parametrant a la main le poids
#               entre la qualite de prediction et le niveau de regularisation.

#QUESTION 2.4.3 : Effectuez la procedure automatique de parametrisation de ce poids, de sorte a ce q'un maximum 
#               de trois variables soit typiquement selectionne et que la qualite de prediction soit optimale.
#               Quelle methode de validation croisee vous semble la plus raisonnable ici ? La selection des 
#               variables est-elle stable ?






