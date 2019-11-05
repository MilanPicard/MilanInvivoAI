##########################################################################

#La première partie du code condiste à importer les données avec les molécules, leur valeur d'activité et les desctipteurs.
#On enlève ensuite les descripteurs corrélés 
#Puis dans la partie création des tableaux, on peut choisir le target sur lequel on a envie de construire le modèle
#Tous le reste du code se fera alors avec ce target
#C'est pas très pratique quand on veux comparer les résultats entre target, mais j'ai pas eu le temps de le changer.

#Du coup, la plupart du code est fait de fonctions qui prennent X (le tableau des molécules avec les descripteurs) comme entrée
#Et la colonne Y (activité pour la cible choisie) reste la même tout au long du code.
#Les résultats mis en commentaire sont les résultats de la cible 1

# Le tableau final avec toutes les activités prédites est en pièce jointe du mail, 
# mais les résultats des modèles pour les autres cibles (2-12) ne sont pas disponibles car beaucoup de travail et d'optimisation ont été fait "à la main".

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import collections
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from yellowbrick.model_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

os.chdir("..")
os.chdir("..")
os.chdir("..")
os.chdir("Users/Milan Picard/Documents/DIDEROT/Internship/Defi")

##########################################################################
#############     Traitement des données / descripteurs      #############
##########################################################################
#Récupération des tableaux
data=pd.read_csv("datadesc.csv")
data_results=data.iloc[:,:12]
data.shape
#On retire les descripteurs qui ont une variance nulle
descriptor_var_nul=list(data.loc[:, data.std() == 0].columns) #Descriptor with variance = 0
data=data.drop(descriptor_var_nul, axis=1) 
data.shape
#On enlève les corrélés seuil>0.75, la regression logistique ne converge pas quand les descripteurs sont trop corrélés, les calculs vont également plus vites.
data_all=data
Cor_variables=[]
for T in range(12):
    Dn=pd.concat([data_all.iloc[:,T],data_all.iloc[:,12:data_all.shape[1]]], axis=1)
    D=Dn.dropna(how='any')
    corD = D.corr().abs()
    upper = corD.where(np.triu(np.ones(corD.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
    for i in to_drop:
        if i not in Cor_variables:
            Cor_variables.append(i)

data=data_all.drop(data_all[Cor_variables], axis=1)





#############             Création des tableaux              #############
#Création du tableau avec target
Target=4

# Avec les descripteurs < 0.75
D_allrows=pd.concat([data.iloc[:,Target-1],data.iloc[:,12:data.shape[1]]], axis=1)
D=D_allrows.dropna(how='any')
Dn=D_allrows.drop(D.index)
target_name = Dn.columns[0]
X = D.loc[:, D.columns != target_name] 
Y = D.loc[:, D.columns == target_name]
#Tableau des molécules sans valeur d'activité qu'il faudra prédire
Xn = Dn.loc[:, D.columns != target_name] 
Yn = Dn.loc[:, D.columns == target_name] #c'est le même Y

# Avec tous les descripteurs, au cas où on veut regarder si un modèle est meilleur avec tous les descripteurs
D_allrows_all=pd.concat([data_all.iloc[:,Target-1],data_all.iloc[:,12:data_all.shape[1]]], axis=1)
D_all=D_allrows_all.dropna(how='any')
Dn_all=D_allrows_all.drop(D_all.index)
X_all = D_all.loc[:, D_all.columns != target_name] 
Y_all = D_all.loc[:, D_all.columns == target_name]
Xn_all = Dn_all.loc[:, D_all.columns != target_name]
Yn_all = Dn_all.loc[:, D_all.columns == target_name]

#############                Data exploration                #############
#Exploration de la colonne d'activité
list01Na=[]
for T in range(12):
    bref=[]
    Dn=pd.concat([data_all.iloc[:,T],data_all.iloc[:,12:data_all.shape[1]]], axis=1)
    D=Dn.dropna(how='any')
    bref.append(D.iloc[:,0].value_counts()[1])
    bref.append(D.iloc[:,0].value_counts()[0])
    bref.append(len(Dn)-len(D))
    bref.append(D.iloc[:,0].value_counts()[0] / D.iloc[:,0].value_counts()[1])
    list01Na.append(bref)

Tableau01Na=pd.DataFrame(list01Na, columns=['1','0','NA','Ratio 0/1'])
print(Tableau01Na)


# PCA
Xscaled_all = StandardScaler().fit_transform(X)
pca = PCA()
PC_all = pca.fit(Xscaled_all)#Scree Plot
variances = np.round(pca.explained_variance_ratio_* 100, decimals=1)
variances15 = variances[0:5]
labels = ['PC' + str(x) for x in range(1, len(variances15)+1)]
plt.bar(x=range(1,len(variances15)+1), height=variances15, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.savefig("ScreePlot")
plt.show()

#Get best features of PC1
dico={}
features_coef=list(abs(PC_all.components_[0]))
for i,coef in enumerate(features_coef):
    dico[list(X.columns)[i]] = coef

dico_sorted=sorted(dico.items(), key=lambda x: x[1], reverse=True)
print(dico_sorted[0:10])

# Exploration de la distribution des molécules en fonction des deux première CPs
def DataVisualisation(target,name):
    DD=pd.concat([data_all.iloc[:,target-1],data_all.iloc[:,12:data_all.shape[1]]], axis=1)
    for i in DD.index:
        if math.isnan(DD.iloc[i,0]):
            DD.iloc[i,0] = 3
    XX = DD.loc[:, DD.columns != name] 
    YY = DD.loc[:, DD.columns == name]
    # PCA
    XXscaled_all = StandardScaler().fit_transform(XX)
    pca = PCA()
    PC_all = pca.fit(XXscaled_all)
        # Distribution des molécules
    pca = PCA()
    XXscaled_all = StandardScaler().fit_transform(XX)
    pca = PCA(n_components=2)
    PC2_all = pca.fit_transform(XXscaled_all)
    PC2_table = pd.DataFrame(data = PC2_all,columns = ['PC1', 'PC2'])
    plt.figure()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("Distribution des molécules du "+name)
    Activity = [0, 3, 1]
    colors = ['g', 'b','r']
    size = [7,10,10]
    transparence=[0.05,0.2,0.8]
    for i in range(3):
        indices = DD[DD[name]==Activity[i]].index
        plt.scatter(PC2_table.loc[indices, 'PC1'], PC2_table.loc[indices, 'PC2'], c = colors[i], s = size[i],alpha=transparence[i])
    plt.legend(['Inactif','NaN','Actif'])
    plt.savefig("Distribution_"+name)
    plt.show()

DataVisualisation(X,target_name)


############################################################################################
##########                                                                        ##########
##########                          Apprentissage des models                      ##########
##########                                                                        ##########
############################################################################################

#############                    PR Courbes                  #############
#La courbe ROC n'est pas très appropriée pour les données déséquilibrés
#La courbe precision recall en fonction du threshold
def courbe (y_test, y_pred_prob, name):
    precision, recall, threshold  = precision_recall_curve(y_test, y_pred_prob)
    rauck = auc(recall, precision)
    plt.figure()
    plt.title("Precision, Recall en fonction du threshold :"+target_name)
    plt.plot(threshold, precision[:-1], "b--", label="Precision")
    plt.plot(threshold, recall[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    plt.savefig(name)
    plt.show()

#Vrai courbe precision-recall, mais je la trouve moins lisible
#    plt.figure()
#    plt.title("Courbe precision-recall")
#    plt.plot(precision, recall, color='b', label='Precision-recall curve (area = %0.2f)' % rauck)
#    plt.xlim([0, 1])
#    plt.ylim([0, 1])
#    plt.xlabel('Recall')
#    plt.ylabel('Precision')
#    plt.legend(loc="best")

#Courbe utilisée pour le model random forest 
#def courbe_forest (yf_test,yf_pred_prob,name):
#    precision, recall, threshold  = precision_recall_curve(yf_test, yf_pred_prob)
#    rauck = auc(recall, precision)
#    plt.figure()
#    plt.title("Precision, Recall en fonction du threshold")
#    plt.plot(threshold, precision[:-1], "b--", label="Precision")
#    plt.plot(threshold, recall[:-1], "g-", label="Recall")
#    plt.ylabel("Score")
#    plt.xlabel("Decision Threshold")
#    plt.legend(loc='best')
#    plt.savefig(name)
#    plt.show()

##########################################################################
######                      Regression Logistique                   ######
##########################################################################

#############              Construction du Model             #############
#Fonction qui calcul une régression logistique sur les données X.
# SMOTE est utilisée pour équilibré le training set.
#Les données sont séparée en 1/5 et 4/5 pour l'apprentissage et la validation
#Lancer la fonction va enregistrer la courbe recall-précision du modèle.
def Resultslog (X,name,threshold=0.5):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/5, random_state=15)
    smo = SMOTE(random_state=15)
    X_train_smo, y_train_smo = smo.fit_sample(X_train, y_train.values.ravel())
    lr = LogisticRegression()
    lr.fit(X_train_smo, y_train_smo.ravel())
    #Prédiction sur le jeu d'apprentissage
    y_predA = lr.predict(X_train_smo)
    ypredA_prob = lr.predict_proba(X_train_smo)[:,1]
    #Boucle permettant de faire varier le threshold
    bi_problistA=[]
    for i in ypredA_prob:
        if i > threshold:
            bi_problistA.append(1)
        elif i <= threshold:
            bi_problistA.append(0)
    #Calcul de la matrice de confuson et des scores associés
    confuA=confusion_matrix(y_train_smo, bi_problistA)
    specificityA = confuA[0,0]/(confuA[0,0]+confuA[0,1])
    recallA = confuA[1,1]/(confuA[1,0]+confuA[1,1])
    precisionA=confuA[1,1]/(confuA[0,1]+confuA[1,1])
    #Prédiction sur le jeu d'évalutation
    y_pred1 = lr.predict(X_test)
    ypred1_prob = lr.predict_proba(X_test)[:,1]
    bi_problist1=[]
    for i in ypred1_prob:
        if i > threshold:
            bi_problist1.append(1)
        elif i <= threshold:
            bi_problist1.append(0)
    confu1=confusion_matrix(y_test, bi_problist1)
    specificity1 = confu1[0,0]/(confu1[0,0]+confu1[0,1])
    recall1 = confu1[1,1]/(confu1[1,0]+confu1[1,1])
    precision1=confu1[1,1]/(confu1[0,1]+confu1[1,1])
    print("Sur échantillon d'apprentissage_smo, avec threshold =",threshold,":")
    print("Specificity: {}\nrecall : {}\nprecision: {}\nConfusion matrix :\n{}".format(specificityA,recallA,precisionA,confuA))
    print("Sur échantillon de test, avec threshold =",threshold,":")
    print("Specificity: {}\nrecall : {}\nprecision: {}\nConfusion matrix :\n{}".format(specificity1,recall1,precision1,confu1))
    #Si on veut voir la courbe associé et l'enregistrer.
    name_curve=str(Y.columns.values[0])+"_Courbes_"+name
    courbe(y_test, ypred1_prob,name_curve)

Resultslog(X,'X')

##########################################################################
#############                 Cross-Validation               #############

#Cross validation implémentée en utilisant les Kfold
def CVlog (X,name,threshold=0.5):
    kf = KFold(n_splits=5, random_state=15, shuffle=True)
    recalls=[]
    precisions=[]
    smo = SMOTE(random_state=15)
    lr = LogisticRegression()
    for trainlist, testlist in kf.split(X,Y):
        bi_problist=[]
        Xtrain, ytrain = X.iloc[list(trainlist)], Y.iloc[list(trainlist)]
        Xtest, ytest = X.iloc[list(testlist)], Y.iloc[list(testlist)]
        Xtrain_smo, ytrain_smo = smo.fit_sample(Xtrain, ytrain.values.ravel())
        lr.fit(Xtrain_smo, ytrain_smo.ravel())
        ypred = lr.predict(Xtest)
        ypred_prob = lr.predict_proba(Xtest)[:,1]
        for i in ypred_prob:
            if i > threshold:
                bi_problist.append(1)
            elif i <= threshold:
                bi_problist.append(0)
        confuK = confusion_matrix(ytest, bi_problist)
        rec=confuK[1,1]/(confuK[1,0]+confuK[1,1])
        pre=confuK[1,1]/(confuK[0,1]+confuK[1,1])
        recalls.append(rec)
        precisions.append(pre)
        print(confuK)
        print(rec)
        print(pre)
        name_curve=str(Y.columns.values[0])+"_CourbesCV_reglog_"+name
        courbe(ytest, ypred_prob,name_curve)
    print("Avec threshold =",threshold,"La moyenne des recalls pendant la cross validation est: {:.3f} (~{:.3f}) ".format(np.mean(recalls), np.std(recalls)))
    print("Avec threshold =",threshold,"La moyenne des precisions pendant la cross validation est: {:.3f} (~{:.3f}) ".format(np.mean(precisions),np.std(precisions)))

CVlog(X,'X')
#X: 
#Avec threshold = 0.5 La moyenne des recalls pendant la cross validation est: 0.589 (~0.053)
#Avec threshold = 0.5 La moyenne des precisions pendant la cross validation est: 0.151 (~0.024)
#CVlog(X,'X',0.81)
#Avec threshold = 0.81 La moyenne des recalls pendant la cross validation est: 0.500 (~0.040)
#Avec threshold = 0.81 La moyenne des precisions pendant la cross validation est: 0.448 (~0.034)

##########################################################################
#############             Réduction de features              #############
# Cette étape peut prendre plusieurs heures (en fonction de l'ordinateur)
lr = LogisticRegression()
rfecv = RFECV(lr, step=1, cv=5, scoring='accuracy')
selector = rfecv.fit(X, Y)
print("The best descriptors after RFECV({}) are: \n {}".format(rfecv.n_features_,X.columns[[index for index, value in enumerate(list(rfecv.support_)) if value == True]]))
X_bestFeatures = X.loc[:,X.columns[[index for index, value in enumerate(list(rfecv.support_)) if value == True]]]

CVlog(X_bestFeatures,'X_bestFeatures')
#Avec threshold = 0.5 La moyenne des recalls pendant la cross validation est: 0.594 (~0.042)
#Avec threshold = 0.5 La moyenne des precisions pendant la cross validation est: 0.149 (~0.026)

##########################################################################
#############          Réduction de Dimensionalité           #############
Xscaled = StandardScaler().fit_transform(X)
pca = PCA()
PC = pca.fit(Xscaled)
X_dimension = pca.fit_transform(Xscaled)
#Avec X_dimension, il faut enlever les .iloc lors de la fabrication des différents sets
def CVlog_dimension (X,name, threshold=0.5):
    kf = KFold(n_splits=5, random_state=15, shuffle=True)
    recalls=[]
    precisions=[]
    smo = SMOTE(random_state=15)
    lr = LogisticRegression()
    for trainlist, testlist in kf.split(X,Y):
        bi_problist=[]
        Xtrain, ytrain = X[list(trainlist)], Y.iloc[list(trainlist)]  #On enlève les iloc sur X(array)
        Xtest, ytest = X[list(testlist)], Y.iloc[list(testlist)]
        Xtrain_smo, ytrain_smo = smo.fit_sample(Xtrain, ytrain.values.ravel())
        lr.fit(Xtrain_smo, ytrain_smo.ravel())
        ypred = lr.predict(Xtest)
        ypred_prob = lr.predict_proba(Xtest)[:,1]
        for i in ypred_prob:
            if i > threshold:
                bi_problist.append(1)
            elif i <= threshold:
                bi_problist.append(0)
        confuK = confusion_matrix(ytest, bi_problist)
        rec=confuK[1,1]/(confuK[1,0]+confuK[1,1])
        pre=confuK[1,1]/(confuK[0,1]+confuK[1,1])
        recalls.append(rec)
        precisions.append(pre)
        print(confuK)
        print(rec)
        print(pre)
        name_curve=str(Y.columns.values[0])+"_CourbesCV_reglog_"+name
        courbe(ytest, ypred_prob,name_curve)
    print("Avec threshold =",threshold,"La moyenne des recalls pendant la cross validation est: {:.3f} (~{:.3f}) ".format(np.mean(recalls), np.std(recalls)))
    print("Avec threshold =",threshold,"La moyenne des precisions pendant la cross validation est: {:.3f} (~{:.3f}) ".format(np.mean(precisions),np.std(precisions)))

CVlog_dimension(X_dimension,'X_dimension')
#Avec threshold = 0.5 La moyenne des recalls pendant la cross validation est: 0.601 (~0.043)
#Avec threshold = 0.5 La moyenne des precisions pendant la cross validation est: 0.145 (~0.027)


##########################################################################
######                          Random Forest                       ######
##########################################################################


#############         Construction du Model Forest          #############

def Resultsforest(X,name,threshold=0.5):
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(X, Y, test_size=1/5, random_state=15)
    forest = RandomForestClassifier(n_estimators=100,random_state=15, class_weight="balanced",max_depth=100)
    forest.fit(Xf_train,yf_train.values.ravel())
    # Finding important features
    Bestfeatures_forest=pd.Series(forest.feature_importances_,index=X.columns).sort_values(ascending=False)
    #list(zip(D1_all.columns[1:], forest.feature_importances_))
    y_predA = forest.predict(Xf_train)
    ypredA_prob = forest.predict_proba(Xf_train)[:,1]
    bi_problistA=[]
    for i in ypredA_prob:
        if i > threshold:
            bi_problistA.append(1)
        elif i <= threshold:
            bi_problistA.append(0)
    confuA=confusion_matrix(yf_train, bi_problistA)
    specificityA = confuA[0,0]/(confuA[0,0]+confuA[0,1])
    recallA = confuA[1,1]/(confuA[1,0]+confuA[1,1])
    precisionA=confuA[1,1]/(confuA[0,1]+confuA[1,1])
    print("Sur échantillon d'apprentissage_smo, avec threshold =",threshold,":")
    print("Specificity: {}\nrecall : {}\nprecision: {}\nConfusion matrix :\n{}".format(specificityA,recallA,precisionA,confuA))
    yf_pred=forest.predict(Xf_test)
    yf_pred_prob = forest.predict_proba(Xf_test)[:,1]
    bi_problist1=[]
    for i in yf_pred_prob:
        if i > threshold:
            bi_problist1.append(1)
        elif i <= threshold:
            bi_problist1.append(0)
    confuf=confusion_matrix(yf_test, bi_problist1)
    specificityf = confuf[0,0]/(confuf[0,0]+confuf[0,1])
    recallf = confuf[1,1]/(confuf[1,0]+confuf[1,1])
    precisionf=confuf[1,1]/(confuf[0,1]+confuf[1,1])
    print("Sur échantillon de test avec threshold =",threshold,":")
    print("Specificity: {}\nrecall : {}\nprecision: {}\nConfusion matrix :\n{}".format(specificityf,recallf,precisionf,confuf))
    name_curve=str(Y.columns.values[0])+"_CourbesForest_Validation"+name
    name_curveA=str(Y.columns.values[0])+"_CourbesForest_Apprentissage"+name
    courbe(yf_test, yf_pred_prob,name_curve)   
    #courbe_forest(yf_train, ypredA_prob,name_curve)   


Resultsforest(X,'X')

#Sur échantillon d'apprentissage_smo, avec threshold = 0.3 :
#Specificity: 0.9973651853152995
#recall : 1.0
#precision: 0.945054945054945
#Confusion matrix :
#[[5678   15]
# [   0  258]]
#Sur échantillon de test avec threshold = 0.3 :
#Specificity: 0.9916434540389972
#recall : 0.5384615384615384
#precision: 0.7
#Confusion matrix :
#[[1424   12]
# [  24   28]]
 
 
##########################################################################
#############                 Cross-Validation               #############

def CVforest (X,name,threshold=0.5):
    kf = KFold(n_splits=5, random_state=15, shuffle=True)
    recalls=[]
    precisions=[]
    forest = RandomForestClassifier(n_estimators=100,random_state=15, 
	class_weight="balanced",
	max_depth=50,
	n_jobs=-1,
	)
    for trainlist, testlist in kf.split(X,Y):
        Xtrain, ytrain = X.iloc[list(trainlist)], Y.iloc[list(trainlist)]
        Xtest, ytest = X.iloc[list(testlist)], Y.iloc[list(testlist)]
        forest.fit(Xtrain,ytrain.values.ravel())
        yf_pred=forest.predict(Xtest)
        yf_pred_prob = forest.predict_proba(Xtest)[:,1]
        bif_problist=[]
        for i in yf_pred_prob:
            if i > threshold:
                bif_problist.append(1)
            elif i <= threshold:
                bif_problist.append(0)
        confuK = confusion_matrix(ytest, bif_problist)
        rec=confuK[1,1]/(confuK[1,0]+confuK[1,1])
        pre=confuK[1,1]/(confuK[0,1]+confuK[1,1])
        recalls.append(rec)
        precisions.append(pre)
        print(confuK)
        print("Recall =",rec)
        print("Precision =",pre)
        name_curve=str(Y.columns.values[0])+"_CourbesCV_Forest_"+name
        courbe(ytest, yf_pred_prob,name_curve)
    print("Avec threshold =",threshold,"La moyenne des recalls pendant la cross validation est: {:.3f} (~{:.3f}) ".format(np.mean(recalls), np.std(recalls)))
    print("Avec threshold =",threshold,"La moyenne des precisions pendant la cross validation est: {:.3f} (~{:.3f}) ".format(np.mean(precisions),np.std(precisions)))


CVforest(X,'X')

#Avec threshold = 0.5 La moyenne des recalls pendant la cross validation est: 0.428 (~0.029)
#Avec threshold = 0.5 La moyenne des precisions pendant la cross validation est: 0.850 (~0.052)

#Avec threshold = 0.7 La moyenne des recalls pendant la cross validation est: 0.371 (~0.027)
#Avec threshold = 0.7 La moyenne des precisions pendant la cross validation est: 0.951 (~0.011)

##########################################################################
######                   Combinaison des models                     ######
##########################################################################
#Random forest sur les prédit actifs de la régression logistic
def Resultscombinaison (X,name,threshold_reg=0.5,threshold_for=0.29):
    PreditPositifs=[]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/5, random_state=15)
    smo = SMOTE(random_state=15)
    X_train_smo, y_train_smo = smo.fit_sample(X_train, y_train.values.ravel())
    lr = LogisticRegression()
    lr.fit(X_train_smo, y_train_smo.ravel())
    y_pred1 = lr.predict(X_test)
    ypred1_prob = lr.predict_proba(X_test)[:,1]
    bi_problist1=[]
    for i in ypred1_prob:
        if i > threshold_reg:
            bi_problist1.append(1)
        elif i <= threshold_reg:
            bi_problist1.append(0)
    confu1=confusion_matrix(y_test, bi_problist1)
    specificity1 = confu1[0,0]/(confu1[0,0]+confu1[0,1])
    recall1 = confu1[1,1]/(confu1[1,0]+confu1[1,1])
    precision1=confu1[1,1]/(confu1[0,1]+confu1[1,1])
    for i,predit in enumerate(bi_problist1):
        if predit == 1:
            PreditPositifs.append(y_test.index[i])	
    Xf = X.loc[PreditPositifs]
    Yf = Y.loc[PreditPositifs]
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(Xf, Yf, test_size=1/5, random_state=11)
    forest = RandomForestClassifier(n_estimators=100,random_state=11, class_weight="balanced",max_depth=100)
    forest.fit(Xf_train,yf_train.values.ravel())
    yf_pred=forest.predict(Xf_test)
    yf_pred_prob = forest.predict_proba(Xf_test)[:,1]
    bi_problist1=[]
    for i in yf_pred_prob:
        if i > threshold_for:
            bi_problist1.append(1)
        elif i <= threshold_for:
            bi_problist1.append(0)
    confuf=confusion_matrix(yf_test, bi_problist1)
    specificityf = confuf[0,0]/(confuf[0,0]+confuf[0,1])
    recallf = confuf[1,1]/(confuf[1,0]+confuf[1,1])
    precisionf=confuf[1,1]/(confuf[0,1]+confuf[1,1])
    print("\n")
    print("Avec regression logistic et threshold =",threshold_reg,":")
    print("Nombre de positifs prédits =",len(PreditPositifs))
    print("Specificity: {}\nrecall : {}\nprecision: {}\nConfusion matrix :\n{}".format(specificity1,recall1,precision1,confu1))
    print("On donne les positifs prédits au Random Forest")
    print("Puis avec random forest, avec threshold de ",threshold_for,":")
    print("Specificity: {}\nrecall : {}\nprecision: {}\nConfusion matrix :\n{}".format(specificityf,recallf,precisionf,confuf))
    name_curve=str(Y.columns.values[0])+"_Courbes_LogisticForest_"+name
    courbe(yf_test, yf_pred_prob,name_curve)   


Resultscombinaison(X,'X')

#Avec regression logistic et threshold = 0.62 :
#Nombre de positifs prédits = 141
#Specificity: 0.9233983286908078
#recall : 0.5961538461538461
#precision: 0.2198581560283688
#Confusion matrix :
#[[1326  110]
# [  21   31]]
#On donne les positifs prédits au Random Forest
#Puis avec random forest, avec threshold de  0.3 :
#Specificity: 0.95
#recall : 1.0
#precision: 0.9
#Confusion matrix :
#[[19  1]
# [ 0  9]]


##########################################################################
#############                 Cross-Validation               #############
#Random forest sur les prédit actifs de la régression logistic
def CVcombinaison (X,threshold_reg=0.6,threshold_for=0.5):
    kf5 = KFold(n_splits=5, random_state=15, shuffle=True)
    kf2 = KFold(n_splits=4, random_state=15, shuffle=True)
    PreditPositifs=[]
    recalls=[]
    precisions=[]
    recalls1=[]
    precisions1=[]
    fold=0
    smo = SMOTE(random_state=15)
    lr = LogisticRegression()
    forest = RandomForestClassifier(n_estimators=100,random_state=15, 
	class_weight="balanced",max_depth=50,n_jobs=-1,)
    for trainlist, testlist in kf5.split(X,Y):
        Xtrain, ytrain = X.iloc[list(trainlist)], Y.iloc[list(trainlist)]
        Xtest, ytest = X.iloc[list(testlist)], Y.iloc[list(testlist)]
        X_train_smo, y_train_smo = smo.fit_sample(Xtrain, ytrain.values.ravel())
        lr.fit(X_train_smo, y_train_smo.ravel())
        y_pred1 = lr.predict(Xtest)
        ypred1_prob = lr.predict_proba(Xtest)[:,1]
        bi_problist1=[]
        for i in ypred1_prob:
            if i > threshold_reg:
                bi_problist1.append(1)
            elif i <= threshold_reg:
                bi_problist1.append(0)
        confu1=confusion_matrix(ytest, bi_problist1)
        recall1 = confu1[1,1]/(confu1[1,0]+confu1[1,1])
        precision1=confu1[1,1]/(confu1[0,1]+confu1[1,1])
        recalls1.append(recall1)
        precisions1.append(precision1)
        for i,predit in enumerate(bi_problist1):
            if predit == 1:
                PreditPositifs.append(ytest.index[i])	
        Xf = X.loc[PreditPositifs]
        Yf = Y.loc[PreditPositifs]
        for trainlistf, testlistf in kf2.split(Xf,Yf):
            Xtrainf, ytrainf = Xf.iloc[list(trainlistf)], Yf.iloc[list(trainlistf)]
            Xtestf, ytestf = Xf.iloc[list(testlistf)], Yf.iloc[list(testlistf)]
            forest.fit(Xtrainf,ytrainf.values.ravel())
            yf_pred=forest.predict(Xtestf)
            yf_pred_prob = forest.predict_proba(Xtestf)[:,1]
            bif_problist=[]
            for i in yf_pred_prob:
                if i > threshold_for:
                    bif_problist.append(1)
                elif i <= threshold_for:
                    bif_problist.append(0)
            confuf = confusion_matrix(ytestf, bif_problist)
            recf=confuf[1,1]/(confuf[1,0]+confuf[1,1])
            pref=confuf[1,1]/(confuf[0,1]+confuf[1,1])
            recalls.append(recf)
            precisions.append(pref)
            fold += 1
            print()
    print()
    print("#####################################################################")
    print()
    print("Avec regression logistic et threshold =",threshold_reg,":")
    print("Nombre de positifs prédits =",len(PreditPositifs))
    print("Mean(precisions) = {:.3f} (~{:.3f}) ".format(np.mean(precisions1),np.std(precisions1)))
    print("Mean(recalls) = {:.3f} (~{:.3f}) ".format(np.mean(recalls1),np.std(recalls1)))
    print()
    print("On donne les positifs prédits au Random Forest")
    print("Avec random forest et threshold =",threshold_for,":")
    print("Sur toute la durée de la cross validation:")
    print("Mean(precisions) = {:.3f} (~{:.3f}) ".format(np.mean(precisions),np.std(precisions)))
    print("Mean(recalls) = {:.3f} (~{:.3f}) ".format(np.mean(recalls),np.std(recalls)))


CVcombinaison(X)

#Avec regression logistic et threshold = 0.62 :
#Nombre de positifs prédits = 709
#Mean(precisions) = 0.237 (~0.039)
#Mean(recalls) = 0.542 (~0.052)
#
#On donne les positifs prédits au Random Forest
#Avec random forest et threshold = 0.3 :
#Sur toute la durée de la cross validation:
#Mean(precisions) = 0.757 (~0.153)
#Mean(recalls) = 0.823 (~0.103)


############################################################################################
##########                                                                        ##########
##########                              Prédiction Finale                         ##########
##########                                                                        ##########
############################################################################################


##########################################################################
######                          Target 1                            ######
##########################################################################


######                        Apprentissage                         ######
# Le random forest a donné les meilleurs résultats
# La prédiction des données d'activité manquantes serait donc fait avec ce model
#La fonction prend X et Y, et ressort le model forest ayant appris
def Apprentissage(X,Y):
    threshold=0.3
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(X, Y, test_size=1/5)
    forest = RandomForestClassifier(n_estimators=100,class_weight="balanced",max_depth=100)
    forest.fit(Xf_train,yf_train.values.ravel())
    yf_pred=forest.predict(Xf_test)
    yf_pred_prob = forest.predict_proba(Xf_test)[:,1]
    bi_problist1=[]
    for i in yf_pred_prob:
        if i > threshold:
            bi_problist1.append(1)
        elif i <= threshold:
            bi_problist1.append(0)
    confuf=confusion_matrix(yf_test, bi_problist1)
    specificityf = confuf[0,0]/(confuf[0,0]+confuf[0,1])
    recallf = confuf[1,1]/(confuf[1,0]+confuf[1,1])
    precisionf=confuf[1,1]/(confuf[0,1]+confuf[1,1])
    print("On donne les positifs prédits au Random Forest")
    print("Puis avec random forest, avec threshold de ",threshold,":")
    print("Specificity: {}\nrecall : {}\nprecision: {}\nConfusion matrix :\n{}".format(specificityf,recallf,precisionf,confuf))
    return(forest)

Model = Apprentissage(X,Y)

#On donne les positifs prédits au Random Forest
#Puis avec random forest, avec threshold de  0.3 :
#Specificity: 0.9916434540389972
#recall : 0.5384615384615384
#precision: 0.7
#Confusion matrix :
#[[1424   12]
# [  24   28]]
#

########################################################################
#######                        Prediction                         ######
#Xn et Yn contiennent les molécules sans valeurs d'activité
threshold = 0.3
yf_predn=Model.predict(Xn)
yf_predn_prob = Model.predict_proba(Xn)[:,1]
Results=[]

for i in yf_predn_prob:
    if i > threshold:
        Results.append(1)
    elif i <= threshold:
        Results.append(0)

nan=0
for i, activity in enumerate(data_results.iloc[:,Target-1]):
    if math.isnan(activity):
        data_results.iloc[i] = Results[nan]
        nan += 1

# Cette prédiction prédit les molécules en utilisant la combinaison de modèle
# #######################################################################
# #######                        Prediction                         ######
# threshold_reg=0.81
# threshold_for=0.53
# y_predn = RegLog.predict(Xn)
# ypredn_prob = RegLog.predict_proba(Xn)[:,1]
# bi_problistn=[]
# for i in ypredn_prob:
#     if i > threshold_reg:
#         bi_problistn.append(1)
#     elif i <= threshold_reg:
#         bi_problistn.append(0)
# 
# postlogistic={}
# for i,predit in enumerate(bi_problistn):
#     postlogistic[Yn.index[i]] = predit 
# 	
# # Tous ce qui a été catégorisé comme 0 ici, est très probable d'être vraiment 0
# # On récupère tous les actifs prédit de la régression logistique
# PreditPositifs=[]
# for i,predit in enumerate(bi_problistn):
#     if predit == 1:
#         PreditPositifs.append(Yn.index[i])	
# 
# # On donne les actifs prédit à la random forest pour qu'elle améliore la précision
# Xnf = Xn.loc[PreditPositifs]
# yf_predn=RandomFor.predict(Xnf)
# yf_predn_prob = RandomFor.predict_proba(Xnf)[:,1]
# bi_problist2n=[]
# for i in yf_predn_prob:
#     if i > threshold_for:
#         bi_problist2n.append(1)
#     elif i <= threshold_for:
#         bi_problist2n.append(0)
# 
# postforest={}
# for i,predit in enumerate(bi_problist2n):
#     postforest[[keys for keys, acti in postlogistic.items() if acti == 1][i]] = predit 
# # On rassemble les informations sous forme de dictionaire
# Resultats={}
# for keys, values in postlogistic.items():
#     if keys in postforest:
#         Resultats[keys] = postforest[keys]
#     else:
#         Resultats[keys] = postlogistic[keys]
# 
# # Remplir la colonne Y (target 1)
# Results1=data.iloc[:,0]
# import math
# row=0
# h=0
# for i in Results1:
#     if math.isnan(i):
#         Results1.iloc[row] = list(Resultats.values())[h]
#         h += 1
#     row += 1
# 
# Results1

#data_results.pd.to_csv(r"Prediction_results.csv",header=True,index=None)