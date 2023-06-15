# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:21:31 2023

@author: TR265290
"""

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import csv

# -----------------------------------------------------------------------------
# Paramètres
# -----------------------------------------------------------------------------
Vi = 20
Vo = 5
Io = 12
Rdson = 7e-3
Fsw = 440e3
L = 2.2e-6
DCR = 8e-3
Vd = 1.7
Vdsch = 860e-3
Dil = (1-Vo/Vi)*Vo/(Fsw*L)
Vgs = 5 # driving voltage
Vsp = 2.2 # switch-point voltage (Miller plateau)
Coss = 383e-12*2 # x2 à 20 V, x1 à 40 V
Csch = 36e-12
Qg = 7.8e-9
Rg = 0.6
Rgext = 2.2
Qgsw = 1.4e-9 # switching gate charge = Qgs + Qgs - Qg(th)
Isource = (Vgs-Vsp)/(Vgs/3.25+Rg+Rgext)
Isink = Vsp/(Vgs/4.25+Rgext)
tr = Qgsw/Isource
tf = Qgsw/Isink
dt = 20e-9

# -----------------------------------------------------------------------------
# Estimation des pertes
# -----------------------------------------------------------------------------
LPonhs = []
LPonls = []
LPswhs = []
LPswls = []
LP = []
LPLr = []
LPdt = []
LPg = []
LEff = []
LPloss = []
LIo = np.linspace(0.1, 12, 500)
for Io in LIo:
    TempSw = 35+4*Io
    StempSw = 1+0.006*TempSw
    TempL = 35*1*Io
    StempL = 1+0.004*TempL
    Ponhs = (Io**2+Dil**2/12)*Rdson*StempSw*Vo/Vi
    Ponls = (Io**2+Dil**2/12)*Rdson*StempSw*(1-Vo/Vi)

    Pswhs = 1/2*Vi*(tr+tf)*Io*Fsw+1/2*Coss*Vi**2*Fsw
    Pswhs = 1/2*Vi*Fsw*(tr*(Io-Dil/2)+tf*(Io+Dil/2))+1/2*Coss*Vi**2*Fsw
    Pswls = 1/2*Vd*Fsw*(tr*(Io-Dil/2)+tf*(Io+Dil/2))+1/2*Csch*Vi**2*Fsw+1/2*Coss*Vi**2*Fsw
    
    PLr = (Io**2+Dil**2/12)*DCR*StempL
    
    Pdt = 2*Vdsch*Io*dt*Fsw
    
    Pg = 2*Qg*Vi*Fsw # Vi car courant tiré sur l'alim du ctrl via le ldo VCC
    
    Ploss = Ponhs+Ponls+Pswhs+Pswls+PLr+Pdt+Pg
    
    LPonhs.append(Ponhs)
    LPonls.append(Ponls)
    LPswhs.append(Pswhs)
    LPswls.append(Pswls)
    LPLr.append(PLr)
    LPdt.append(Pdt)
    LPg.append(Pg)
    LP.append(Vo*Io)
    LPloss.append(Ploss)
    LEff.append((Vo*Io)/(Vo*Io+Ploss))

# -----------------------------------------------------------------------------
# Affichage des pertes estimées
# -----------------------------------------------------------------------------
plt.figure(1)
plt.plot(LIo,LPonhs,color='r')
plt.plot(LIo,LPonls,color='b')
plt.plot(LIo,LPswhs,color='r',linestyle='--')
plt.plot(LIo,LPswls,color='b',linestyle='--')
plt.plot(LIo,LPLr,color='y',linestyle='-')
plt.plot(LIo,LPdt,color='m',linestyle='-')
plt.plot(LIo,LPg,color='m',linestyle='--')
# plt.plot(range(0,12),LP,color='g')
plt.legend(["Ponhs","Ponls","Pswhs","Pswls","PLr","Pdt","Pg"])
plt.grid()
plt.figure(2)
plt.plot(LIo,LPloss,color='r')
plt.title("Total power losses")
plt.grid()

# -----------------------------------------------------------------------------
# Extraction des données expérimentales
# -----------------------------------------------------------------------------
def extract_columns(csv_file):
    # Initialisation des listes pour stocker les colonnes et les labels
    columns = []

    # Ouverture du fichier CSV en mode lecture
    with open(csv_file, 'r') as file:
        # Création d'un objet lecteur CSV
        csv_reader = csv.reader(file,delimiter=';')

        # Lecture de la première ligne pour déterminer le nombre de colonnes
        header = next(csv_reader)
        num_columns = len(header)

        # Initialisation des listes pour chaque colonne
        for _ in range(num_columns):
            columns.append([])
        
        for row in csv_reader:
            for i in range(num_columns):
                columns[i].append(float(row[i]))

    return header, columns

# Exemple d'utilisation
csv_file = 'Polaris_V2_tests_results_sansdiode.csv'
labels, columns = extract_columns(csv_file)

# Affichage des labels
# print("Labels:")
# print(labels)
# print()

# # Affichage des données extraites
# for i, column in enumerate(columns):
#     print(f'Colonne {i+1}:')
#     print(column)
#     print()

plt.figure(3)
plt.plot(LIo, LEff)
plt.plot(columns[2][8:16], columns[3][8:16])
plt.grid()
plt.ylim([0.7, 1])
plt.xlabel("Iout (A)")
plt.ylabel("Rendement")
plt.legend(["Estimatation","Experimental"])


# -----------------------------------------------------------------------------
# Stacked area chart
# -----------------------------------------------------------------------------

# Make data
data = pd.DataFrame({'LIo':LIo, 'LPonhs':np.divide(LPonhs,np.add(LP,LPloss)), 'LPonls':np.divide(LPonls,np.add(LP,LPloss)), 'LPswhs':np.divide(LPswhs,np.add(LP,LPloss)), 'LPswls':np.divide(LPswls,np.add(LP,LPloss)), 'LPLr':np.divide(LPLr,np.add(LP,LPloss)), 'LPdt':np.divide(LPdt,np.add(LP,LPloss)), 'LPg':np.divide(LPg,np.add(LP,LPloss)), 'LP':LEff})
# sns.set_theme()
# sns.set_style("dark")

# Make the plot
plt.figure(4)
plt.stackplot(data["LIo"], data["LP"], data["LPonls"],data["LPswls"],data["LPonhs"],data["LPswhs"],data["LPLr"],data["LPdt"],data["LPg"], labels =["P","Ponls","Pswls","Ponhs","Pswhs","PLr","Pdt","Pg"])
plt.legend(loc='lower right')
plt.margins(0,0)
plt.title('100 % stacked area chart')
plt.ylim([0.9, 1])
plt.xlim(LIo[0], LIo[-1])
plt.grid()
plt.plot(LIo,LEff, color='k',linestyle='-', linewidth=2)
plt.show()
