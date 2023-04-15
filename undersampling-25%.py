import numpy as np #cálculos matemáticos
import pandas as pd #criar os dataframes
import matplotlib.pyplot as plt #criar gráficos
from sklearn.preprocessing import StandardScaler #padronizar dados
from sklearn.model_selection import train_test_split #dividir dados em treino e teste

df = pd.read_csv('creditcard.csv') #importar os dados para o notebook numa variável chamada 'df'
df.head() #verificar o inicio dos dados

df.describe() #estatística dos dados

df.isnull().sum().max() #verificar se há dados faltantes

df.columns #verificar as colunas do dataframe

#criar gráfico para conhecer o balanceamento das fraudes
contagem_classes = pd.value_counts(df['Class'], sort = True).sort_index() #contagem de fraudes e não fraudes
contagem_classes.plot(kind = 'bar') #criação do histograma de barras
plt.title("Histograma das Fraudes") #título do histograma
plt.xlabel("Fraudes") #título do eixo X do histograma
plt.ylabel("Frequencia") #título do eixo Y do histograma

#saber o percentual exato de fraudes e não fraudes
print("Não Fraudes: ",df['Class'].value_counts()[0]/len(df)*100,'%')
print("Fraudes: ",df['Class'].value_counts()[1]/len(df)*100,'%')
#imprimir o somatório de 0 e 1 na coluna classe dividido pela quantidade de valores totais

df['Preço'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1)) #padronizar a coluna Amount entre -1 e 1
df['Tempo'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1,1)) #padronizar a coluna Time entre -1 e 1
Preço = df['Preço'] #criar variável da coluna Preço
Tempo = df['Tempo'] #criar variável da coluna Tempo
df.drop(['Time','Amount'], axis=1, inplace=True) #deletar as colunas Time e Amount
df.drop(['Preço', 'Tempo'], axis=1, inplace=True) #deletar as colunas Preço e Tempo pois elas foram adicionadas ao final do dataframe com a função anterior
df.insert(0, 'Preço', Preço) #inserir a variável coluna Preço na posição 0
df.insert(1, 'Tempo', Tempo) #inserir a variável coluna Tempo na posição 1

X = df.iloc[:, df.columns != 'Class'] #criar uma matriz X sem a coluna de fraudes
y = df.iloc[:, df.columns == 'Class'] #criar uma matriz y apenas com a coluna de fraudes

numero_fraudes = len(df[df.Class == 1]) #criar uma variável que diga o número de fraudes
indice_fraudes = df[df.Class == 1].index #criar variável com índice das fraudes

indice_normal = df[df.Class == 0].index #criar variável com índice das não fraudes

aleatorio_indice_normal = np.random.choice(indice_normal, 1476, replace = False) #criar variável que contenha o quantidade de não fraudes igual ao de fraude de forma aleatória

aleatorio_indice_normal = np.array(aleatorio_indice_normal) #criar variável que contenha a matriz do índice escolhido na função anterior

undersample_indices = np.concatenate([indice_fraudes, aleatorio_indice_normal]) #concatenar os índices de fraude e não fraude

undersample_df = df.iloc[undersample_indices, :] #criar variável que contenha a matriz completa das fraudes e não fraudes acima

X_undersample = undersample_df.iloc[:, undersample_df.columns != 'Class'] #matriz do undersample sem a supervisão
y_undersample = undersample_df.iloc[:, undersample_df.columns == 'Class'] #matriz do undersample apenas com a supervisão de fraudes

print("Percentual de transações genuínas: ", len(undersample_df[undersample_df.Class == 0])/len(undersample_df)*100,"%")
print("Percentual de fraudes: ", len(undersample_df[undersample_df.Class == 1])/len(undersample_df)*100,"%")
print("Número de transações do undersample: ", len(undersample_df))
#verificar se o undersample está balanceado

contagem_undersample = pd.value_counts(undersample_df['Class'], sort = True).sort_index() #contagem de fraudes e não fraudes
contagem_undersample.plot(kind = 'bar') #criação do histograma de barras
plt.title("Histograma das Fraudes") #título do histograma
plt.xlabel("Fraudes") #título do eixo X do histograma
plt.ylabel("Frequencia") #título do eixo Y do histograma

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0) #divisão de todo dataset em treino e teste 20%

print("Número de transações no dataset de treino: ", len(X_train))
print("Número de transações no dataset de teste: ", len(X_test))
print("Total de transações: ", len(X_train)+len(X_test))

X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample, y_undersample, test_size = 0.2, random_state = 0)
#divisão do subdataset em treino e teste 20%
print("Número de transações no undersample de treino: ", len(X_train_undersample))
print("Número de transações no undersample de teste: ", len(X_test_undersample))
print("Total de transações: ", len(X_train_undersample)+len(X_test_undersample))


"""# Regressão Logística"""

from sklearn.linear_model import LogisticRegression

regressao_logistica = LogisticRegression(random_state=0, C = 1).fit(X_train_undersample, y_train_undersample.values.ravel())

y_pred_RL = regressao_logistica.predict(X)

import seaborn as sns
from sklearn.metrics import confusion_matrix

cm_RL = confusion_matrix(y, y_pred_RL)
f_RL = sns.heatmap(cm_RL, annot=True, fmt='d')

from sklearn.metrics import confusion_matrix
confusion_matrix(y,y_pred_RL)

from sklearn.metrics import precision_score

precision_score(y, y_pred_RL, average = 'binary')

from sklearn.metrics import recall_score

recall_score(y, y_pred_RL, average = 'binary')

from sklearn.metrics import accuracy_score

accuracy_score(y, y_pred_RL)

from sklearn.metrics import f1_score
f1_score(y, y_pred_RL, average = 'binary')

from sklearn.metrics import fbeta_score
fbeta_score(y, y_pred_RL, beta=2)

from sklearn.metrics import classification_report
print('Logistic Regression:')
print(classification_report(y, y_pred_RL, digits = 5))

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y, y_pred_RL)


"""# KNN"""

from sklearn.neighbors import KNeighborsClassifier
KNeighbors = KNeighborsClassifier(n_neighbors=2).fit(X_train_undersample, y_train_undersample.values.ravel())

y_pred_KNN = KNeighbors.predict(X)

cm_KNN = confusion_matrix(y, y_pred_KNN)
f_KNN = sns.heatmap(cm_KNN, annot=True, fmt='d')

confusion_matrix(y,y_pred_KNN)

precision_score(y, y_pred_KNN, average = 'binary')

accuracy_score(y, y_pred_KNN)

f1_score(y, y_pred_KNN, average = 'binary')

fbeta_score(y, y_pred_KNN, beta=2)

recall_score(y, y_pred_KNN, average = 'binary')

print('KNN:')
print(classification_report(y, y_pred_KNN, digits = 5))

precision_recall_fscore_support(y, y_pred_KNN)


"""# Support Vector Machines"""

from sklearn import svm
SVM = svm.SVC()
SVM.fit(X_train_undersample, y_train_undersample.values.ravel())

y_pred_SVM = SVM.predict(X)

cm_SVM = confusion_matrix(y, y_pred_SVM)
f_SVM = sns.heatmap(cm_SVM, annot=True, fmt='d')

confusion_matrix(y,y_pred_SVM)

precision_score(y, y_pred_SVM, average = 'binary')

accuracy_score(y, y_pred_SVM)

f1_score(y, y_pred_SVM, average = 'binary')

fbeta_score(y, y_pred_SVM, beta=2)

recall_score(y, y_pred_SVM, average = 'binary')

print('SVM:')
print(classification_report(y, y_pred_SVM, digits = 5))

precision_recall_fscore_support(y, y_pred_SVM)


"""# Decision Tree"""

from sklearn import tree
TREE = tree.DecisionTreeClassifier()
TREE.fit(X_train_undersample, y_train_undersample.values.ravel())

y_pred_TREE = TREE.predict(X)

cm_TREE = confusion_matrix(y, y_pred_TREE)
f_TREE = sns.heatmap(cm_TREE, annot=True, fmt='d')

confusion_matrix(y,y_pred_TREE)

precision_score(y, y_pred_TREE, average = 'binary')

accuracy_score(y, y_pred_TREE)

f1_score(y, y_pred_TREE, average = 'binary')

fbeta_score(y, y_pred_TREE, beta=2)

recall_score(y, y_pred_TREE, average = 'binary')

print('TREE:')
print(classification_report(y, y_pred_TREE, digits = 5))

precision_recall_fscore_support(y, y_pred_TREE)


"""# Random Forests"""

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=50)
RF.fit(X_train_undersample, y_train_undersample.values.ravel())

y_pred_RF = RF.predict(X)

cm_RF = confusion_matrix(y, y_pred_RF)
f_RF = sns.heatmap(cm_RF, annot=True, fmt='d')

confusion_matrix(y,y_pred_RF)

precision_score(y, y_pred_RF, average = 'binary')

accuracy_score(y, y_pred_RF)

f1_score(y, y_pred_RF, average = 'binary')

fbeta_score(y, y_pred_RF, beta=2)

recall_score(y, y_pred_RF, average = 'binary')

print('RANDOM FORESTS:')
print(classification_report(y, y_pred_RF, digits = 5))

precision_recall_fscore_support(y, y_pred_RF)


"""# Naive Bayes"""

from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train_undersample, y_train_undersample.values.ravel())

y_pred_NB = NB.predict(X)

cm_NB = confusion_matrix(y, y_pred_NB)
f_NB = sns.heatmap(cm_NB, annot=True, fmt='d')

confusion_matrix(y,y_pred_NB)

precision_score(y, y_pred_NB, average = 'binary')

accuracy_score(y, y_pred_NB)

f1_score(y, y_pred_NB, average = 'binary')

fbeta_score(y, y_pred_NB, beta=2)

recall_score(y, y_pred_NB, average = 'binary')

print('NAIVE BAYES:')
print(classification_report(y, y_pred_NB, digits = 5))

precision_recall_fscore_support(y, y_pred_NB)


"""# Multi-layer Perceptron"""

from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=500)
MLP.fit(X_train_undersample, y_train_undersample.values.ravel())

y_pred_MLP = MLP.predict(X)

cm_MLP = confusion_matrix(y, y_pred_MLP)
f_MLP = sns.heatmap(cm_MLP, annot=True, fmt='d')

confusion_matrix(y,y_pred_MLP)

precision_score(y, y_pred_MLP, average = 'binary')

accuracy_score(y, y_pred_MLP)

f1_score(y, y_pred_MLP, average = 'binary')

fbeta_score(y, y_pred_MLP, beta=2)

recall_score(y, y_pred_MLP, average = 'binary')

print('Multi-layer Perceptron:')
print(classification_report(y, y_pred_MLP, digits = 5))

precision_recall_fscore_support(y, y_pred_MLP)


"""# Consolidação"""

Resultados = {'Regressão Logística': round(accuracy_score(y, y_pred_RL),8),
              'KNN':                 round(accuracy_score(y, y_pred_KNN),8),
              'SVM':                 round(accuracy_score(y, y_pred_SVM),8),
              'Decision Tree':       round(accuracy_score(y, y_pred_TREE),8),
              'Random Forests':      round(accuracy_score(y, y_pred_RF),8),
              'Naive Bayes':         round(accuracy_score(y, y_pred_NB),8),
              'MLP':                 round(accuracy_score(y, y_pred_MLP),8)}

for i in sorted(Resultados, key = Resultados.get, reverse = True):
      print('Accuracy',i, Resultados[i])

Resultados = {'Regressão Logística': round(recall_score(y, y_pred_RL, average = 'binary'),8),
              'KNN':                 round(recall_score(y, y_pred_KNN, average = 'binary'),8),
              'SVM':                 round(recall_score(y, y_pred_SVM, average = 'binary'),8),
              'Decision Tree':       round(recall_score(y, y_pred_TREE, average = 'binary'),8),
              'Random Forests':      round(recall_score(y, y_pred_RF, average = 'binary'),8),
              'Naive Bayes':         round(recall_score(y, y_pred_NB, average = 'binary'),8),
              'MLP':                 round(recall_score(y, y_pred_MLP, average = 'binary'),8)}

for i in sorted(Resultados, key = Resultados.get, reverse = True):
      print('Recall',i, Resultados[i])

Resultados = {'Regressão Logística': round(precision_score(y, y_pred_RL, average = 'binary'),8),
              'KNN':                 round(precision_score(y, y_pred_KNN, average = 'binary'),8),
              'SVM':                 round(precision_score(y, y_pred_SVM, average = 'binary'),8),
              'Decision Tree':       round(precision_score(y, y_pred_TREE, average = 'binary'),8),
              'Random Forests':      round(precision_score(y, y_pred_RF, average = 'binary'),8),
              'Naive Bayes':         round(precision_score(y, y_pred_NB, average = 'binary'),8),
              'MLP':                 round(precision_score(y, y_pred_MLP, average = 'binary'),8)}

for i in sorted(Resultados, key = Resultados.get, reverse = True):
      print('Precision',i, Resultados[i])

Resultados = {'Regressão Logística': round(f1_score(y, y_pred_RL, average = 'binary'),8),
              'KNN':                 round(f1_score(y, y_pred_KNN, average = 'binary'),8),
              'SVM':                 round(f1_score(y, y_pred_SVM, average = 'binary'),8),
              'Decision Tree':       round(f1_score(y, y_pred_TREE, average = 'binary'),8),
              'Random Forests':      round(f1_score(y, y_pred_RF, average = 'binary'),8),
              'Naive Bayes':         round(f1_score(y, y_pred_NB, average = 'binary'),8),
              'MLP':                 round(f1_score(y, y_pred_MLP, average = 'binary'),8)}

for i in sorted(Resultados, key = Resultados.get, reverse = True):
      print('F1',i, Resultados[i])

Resultados = {'Regressão Logística': round(fbeta_score(y, y_pred_RL, average = 'binary', beta=2),8),
              'KNN':                 round(fbeta_score(y, y_pred_KNN, average = 'binary', beta=2),8),
              'SVM':                 round(fbeta_score(y, y_pred_SVM, average = 'binary', beta=2),8),
              'Decision Tree':       round(fbeta_score(y, y_pred_TREE, average = 'binary', beta=2),8),
              'Random Forests':      round(fbeta_score(y, y_pred_RF, average = 'binary', beta=2),8),
              'Naive Bayes':         round(fbeta_score(y, y_pred_NB, average = 'binary', beta=2),8),
              'MLP':                 round(fbeta_score(y, y_pred_MLP, average = 'binary', beta=2),8)}

for i in sorted(Resultados, key = Resultados.get, reverse = True):
      print('F2',i, Resultados[i])