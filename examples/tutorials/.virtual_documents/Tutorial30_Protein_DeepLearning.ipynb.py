get_ipython().getoutput("curl -Lo conda_installer.py https://raw.githubusercontent.com/deepchem/deepchem/master/scripts/colab_install.py")
import conda_installer
conda_installer.install()
get_ipython().getoutput("/root/miniconda/bin/conda info -e")


get_ipython().getoutput("pip install --pre deepchem")


get_ipython().getoutput("pip install propy3 ")


#imports 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


data =pd.read_csv("./all_measurementsProtherDB.tsv",delimiter="\t")


data.shape


df = data[ (data['MEASURE'] == 'DSC') & (data['MUTATION'] == 'wild-type')]


df.shape


no_mutants_dataset = df.drop_duplicates(['PDB_wild'])


no_mutants_dataset.shape


df2 = data[ (data['MEASURE'] == 'DSC')]


df2.shape


no_wild_type_repeated  =df2.drop_duplicates(['MUTATION','PDB_wild'],keep='first')


no_wild_type_repeated.sort_values('UniProt_ID')[['UniProt_ID','PDB_wild','PROTEIN']]


Mutation_list = no_wild_type_repeated[['UniProt_ID','MUTATION']].set_index('UniProt_ID')


Mutation_list


import requests, sys
def Seq_From_AccNum(num):
    '''
    perform a http  request to the EBI-API to obtain the sequence based on the Uniprot Accession number
    return a string 
    
    '''
    requestURL = "https://www.ebi.ac.uk/proteins/api/features/{}".format(num)
    r = requests.get(requestURL, headers={ "Accept" : "application/json"})
    if not r.ok:
        print("Failure in Uniprot request")
        return None
        r.raise_for_status()
        sys.exit()
    responseBody = r.json()
    return responseBody['sequence']


import re
def MutateSeq(seq,Mutant):
    '''
    mutate a sequence based on a string (Mutant) that has the notation : 
    A###B where A is the wildtype aminoacid ### the position and B the mutation
    
    '''
    aalist = re.findall('([A-Z])([0-9]+)([A-Z])', Mutant)
    
    #(len(aalist)==1):
    newseq=seq
    listseq=list(newseq)
    for aas in aalist:
        wildAA = aas[0]
        pos = int(aas[1]) -1
        if(pos >= len(listseq)):
            print("Mutation not in the range of the protein")
            return None
        MutAA = aas[-1]
        
        if(listseq[pos]==wildAA):
            
            listseq[pos]=MutAA
            
        else:
            print("WildType AA does not match")
            return None
    return("".join(listseq))



Mutation_list.sort_values('UniProt_ID',inplace=True)


Mutation_list


import time 
t0 = time.time()
Sequences = {}
fail  = {}
AccNum = []
count = 0 
print("Perfoming data curation : ")

for accnumber, row in Mutation_list.iterrows():
    #if(count == 100):
    #    break
    #count += 1
    mutation = row['MUTATION'].split("(")[0].strip()
    print("{} - {}".format(accnumber,mutation), end =" ")
    name ="{}-{}".format(accnumber,mutation)
    if(accnumber=='-'):
        fail[accnumber] = [mutation]
        continue
    if accnumber not in AccNum:
        AccNum.append(accnumber)
        seq = Seq_From_AccNum(accnumber)
        if(seq == None):
            continue
    if(mutation =='wild-type'):
        name ="{}-{}".format(accnumber,"WT")
        Sequences[name]=seq
    else:
        mutseq = MutateSeq(seq,mutation)
        if(mutseq==None):
            if(accnumber not in fail ):
                fail[accnumber] = [mutation]
            else:
                fail[accnumber].append(mutation)
        Sequences[name] = mutseq

print("Total time analyzing all the dataFrame {} s".format(time.time() - t0))


fail


temp_accnum = no_wild_type_repeated[['UniProt_ID','MUTATION','Tm_(C)','PDB_wild']]
arr_acc_temp = temp_accnum.to_numpy()

temp_dic={}

for i in arr_acc_temp:
    acc = i[0]
    if(i[1] == 'wild-type'):
        mut ='WT'
    else:
        mut = i[1].split('(')[0].strip()

    name="{}-{}".format(acc,mut)
    temp_dic[name]=[i[-2],i[-1].upper()]
    
with open('sequences_protherm.csv','w') as file:
    file.write("AccNumber,T_m,PDBID,Sequence \n")
    for k,v in Sequences.items():
        if(v==None):
            continue
        temp = temp_dic[k][0]
        code_pdb = temp_dic[k][1]


        text = "{},{},{},{} \n".format(k,temp,code_pdb,v)
        file.write(text)


import deepchem as dc
from rdkit import Chem
import pandas as pd 


Final_Data  = pd.read_csv("sequences_protherm.csv")


seq_list = list(Final_Data['Sequence '])


codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


max_seq= 0 
for i in seq_list:
    if(len(i)>max_seq):
        max_seq=len(i)
max_seq


OneHotFeaturizer = dc.feat.OneHotFeaturizer(codes,max_length=max_seq)


features = OneHotFeaturizer.featurize(seq_list)


temp = [float(x.split("(")[0]) for x in list(Final_Data['T_m'])]


dc_dataset = dc.data.NumpyDataset(X=features,y=temp)


dc_dataset.X.shape


from deepchem import splits
splitter = splits.RandomSplitter()
train, test  = splitter.train_test_split(dc_dataset,seed=42)


dc_dataset.X.shape[1:]


from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(units=32, activation='relu', input_shape=dc_dataset.X.shape[1:]),
    layers.Dropout(0.2),
    layers.Dense(units=32, activation='relu'), 
    layers.Dropout(0.2),
    layers.Dense(units=32, activation='relu'), 
    layers.Dropout(0.2),
    layers.Dense(units=1),
])
model.compile(loss='mae', optimizer='adam')

print(model.summary())

history = model.fit(
    train.X, train.y,
    validation_data=(test.X,test.y),
    batch_size=24,
    epochs=10,
)

## perform a plot of loss vs epochs 
import matplotlib.pyplot as plt
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()


dc_model = dc.models.KerasModel(model,dc.metrics.mae_score)


from deepchem import splits
splitter = splits.RandomSplitter()
train, test  = splitter.train_test_split(dc_dataset,seed=42)


train.X.shape


from sklearn.ensemble import RandomForestRegressor
from deepchem.utils.evaluate import Evaluator
import pandas as pd
seed = 42 # Set a random seed to get stable results
sklearn_model = RandomForestRegressor(n_estimators=100, max_features='sqrt')
sklearn_model.random_state = seed
model = dc.models.SklearnModel(sklearn_model)
model.fit(train)


from propy import PyPro


aaComplist = []
CTDList =[]
for seq in seq_list:
    Obj = PyPro.GetProDes(seq)
    aaComplist.append(np.array(list(Obj.GetAAComp().values())))
    CTDList.append(np.array(list(Obj.GetCTD().values())))


dc_dataset_aacomp = dc.data.NumpyDataset(X=aaComplist,y= temp)
dc_dataset_ctd = dc.data.NumpyDataset(X=CTDList,y= temp)


from deepchem import splits
splitter = splits.RandomSplitter()
train, test  = splitter.train_test_split(dc_dataset_aacomp,seed=42)
from sklearn.ensemble import RandomForestRegressor
from deepchem.utils.evaluate import Evaluator
import pandas as pd
print("RandomForestRegressor")
seed = 42 # Set a random seed to get stable results
sklearn_model = RandomForestRegressor(n_estimators=100, max_features='sqrt')
sklearn_model.random_state = seed
model = dc.models.SklearnModel(sklearn_model)
model.fit(train)
metric = dc.metrics.Metric(dc.metrics.mae_score)
train_score = model.evaluate(train, [metric])
test_score = model.evaluate(test, [metric])
print("Train score is : {}".format(train_score))
print("Test score is : {}".format(test_score))
print("SupportVectorMachineRegressor")
from sklearn.svm import SVR
svr_sklearn = SVR(kernel="poly",degree=4)
svr_sklearn.random_state = seed 
model = dc.models.SklearnModel(svr_sklearn)
model.fit(train)
metric = dc.metrics.Metric(dc.metrics.mae_score)
train_score = model.evaluate(train, [metric])
test_score = model.evaluate(test, [metric])
print("Train score is : {}".format(train_score))
print("Test score is : {}".format(test_score))


from deepchem import splits
splitter = splits.RandomSplitter()
train, test  = splitter.train_test_split(dc_dataset_ctd,seed=42)
from sklearn.ensemble import RandomForestRegressor
from deepchem.utils.evaluate import Evaluator
import pandas as pd
print("RandomForestRegressor")
seed = 42 # Set a random seed to get stable results
sklearn_model = RandomForestRegressor(n_estimators=100, max_features='sqrt')
sklearn_model.random_state = seed
model = dc.models.SklearnModel(sklearn_model)
model.fit(train)
metric = dc.metrics.Metric(dc.metrics.mae_score)
train_score = model.evaluate(train, [metric])
test_score = model.evaluate(test, [metric])
print("Train score is : {}".format(train_score))
print("Test score is : {}".format(test_score))
print("SupportVectorMachineRegressor")
from sklearn.svm import SVR
svr_sklearn = SVR(kernel="poly",degree=4)
svr_sklearn.random_state = seed 
model = dc.models.SklearnModel(svr_sklearn)
model.fit(train)
metric = dc.metrics.Metric(dc.metrics.mae_score)
train_score = model.evaluate(train, [metric])
test_score = model.evaluate(test, [metric])
print("Train score is : {}".format(train_score))
print("Test score is : {}".format(test_score))
