from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
from mlm import MinimalLearningMachine as MLM
from mlm.selectors import KSSelection, NLSelection
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error





def preprocessing(dataset):
    # séparer les colonnes en deux types catégorielles et numériques
    cat_col=[col for col in dataset.columns if dataset[col].dtype=='object']
    num_col=[col for col in dataset.columns if dataset[col].dtype=='int64' or dataset[col].dtype=='float64']
    # garder uniquement les colonnes catégorielles et supprimer les valeurs manquantes 
    dataset = dataset[num_col].dropna(axis=0)
    # 10 pérmutation aléatoire pour mélanger les données 
    df_shuffled=dataset.sample(frac=1).reset_index(drop=True)
    #for i in range(10):
     #   df_shuffled=df_shuffled.sample(frac=1).reset_index(drop=True)
    # centrer et réduire les données
    data = StandardScaler().fit_transform(df_shuffled)
    # diviser les données (entrée, sortie)
    size = data.shape[1]
    X = data[:,:size-1]
    Y = data[:,size-1]
    return X,Y

domain = pd.read_csv("Abalone/abalone.domain",delimiter=":", names=["column","type" ])# Pour charger les noms des dolonnes
abalone = pd.read_csv("Abalone/abalone.data",names=domain.column.to_list()) # charher la dataset, 
X,y = preprocessing(abalone)


y = y.reshape((len(y),1))
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1/3,random_state=42)

mlm1 = MLM()
mlm1.fit(x_train, y_train)
ypred1 = mlm1.predict(x_test)

mlm2 = MLM(selector=KSSelection())
mlm2.fit(x_train, y_train)
ypred2 = mlm2.predict(x_test)

mlm3 = MLM(selector=NLSelection())
mlm3.fit(x_train, y_train)
ypred3 = mlm3.predict(x_test)

print("With mlm1 :",mean_squared_error(ypred1,y_test))
print("With mlm2 :",mean_squared_error(ypred2,y_test))
print("With mlm3 :",mean_squared_error(ypred3,y_test))


