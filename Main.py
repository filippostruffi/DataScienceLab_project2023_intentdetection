import librosa
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataframe_development = pd.read_csv("dsl_data/development.csv",index_col=0)
dataframe_evaluation = pd.read_csv("dsl_data/evaluation.csv")
develop_set = [] 
eval_set = []
develop_label = []
sampling_frequency = 22050 
maximal_length = 279552 
hop_length = 512

# Construction of development set and extraction of labels

count = 0
for a, row in dataframe_development.iterrows():
    path = row['path']
    intent = row['action'] + row['object']
    develop_label.append(intent)
    data, sr = librosa.load(f"{path}")
    data, ind = librosa.effects.trim(data, top_db=30)
    audio = np.pad(data, (0, maximal_length - len(data)), 'constant', constant_values=0)
    mfccs = librosa.feature.mfcc(y=audio, sr = sampling_frequency, n_mfcc=13, hop_length=512)
    MFCC_extend = np.ravel(mfccs,order='F')
    stft = np.abs(librosa.stft(audio.astype(float))) ** 2
    spectral_energy = np.sum(stft, axis=0)
    plp = librosa.feature.melspectrogram(y=audio.astype(float), sr = sampling_frequency, n_mels=13, n_fft=2048, hop_length=512, power=2.0)
    plp = librosa.power_to_db(plp, ref=np.max)
    PLP = np.ravel(plp,order='F')
    features = np.concatenate((MFCC_extend,spectral_energy,PLP),axis=0)
    develop_set.append(features)
    count = count + 1
    print(count)

develop_set = np.array(develop_set)
min_max_scaler = MinMaxScaler()
develop_set = min_max_scaler.fit_transform(develop_set)

# Construction of evaluation set

count = 0
for a, row in dataframe_evaluation.iterrows():
    path = row['path']
    data, sr = librosa.load(f"{path}")
    sampling_frequency = sr
    data, ind = librosa.effects.trim(data, top_db=30)
    audio = np.pad(data, (0, maximal_length - len(data)), 'constant', constant_values=0)
    mfccs = librosa.feature.mfcc(y=audio, sr=sampling_frequency, n_mfcc=13, hop_length=512)
    MFCC_extend=np.ravel(mfccs,order='F')
    stft = np.abs(librosa.stft(audio.astype(float))) ** 2
    spectral_energy = np.sum(stft, axis=0)
    plp = librosa.feature.melspectrogram(y=audio.astype(float), sr=sampling_frequency, n_mels=13, n_fft=2048, hop_length=512, power=2.0)
    plp = librosa.power_to_db(plp, ref=np.max)
    PLP = np.ravel(plp,order='F')
    features = np.concatenate((MFCC_extend,spectral_energy,PLP),axis=0)
    eval_set.append(features)
    count = count + 1
    print(count)
    
eval_set = np.array(eval_set)
min_max_scaler = MinMaxScaler()
eval_set = min_max_scaler.fit_transform(eval_set)


# PCA for extraction of set of most meaningful features

print('START PCA')
n=60
pca = PCA(n_components=n)   

inter=pca.fit_transform(np.concatenate((develop_set,eval_set),axis=0))
develop_set = inter[:len(develop_set)]
eval_set = inter[len(develop_set):]

# Construction of model

print('START MODEL')
X_train, X_test, y_train, y_test = train_test_split(develop_set, np.array(develop_label), test_size=0.2)
clf = RandomForestClassifier(n_estimators=1500)
clf.fit(X_train, y_train)
ypred = clf.predict(X_test)
print(f"accuracy on development: {accuracy_score(y_test, np.array(ypred)) * 100}%")

# Classification of evaluation test with CSV file export

y_pred_eval = clf.predict(eval_set)

eval_fin = pd.DataFrame(y_pred_eval)
eval_fin = eval_fin.rename(columns={0: 'Predicted'})
eval_fin["Id"] = dataframe_evaluation["Id"]
eval_fin = eval_fin.reindex(columns=['Id','Predicted'])
eval_fin.to_csv("Prediction.csv",index = False)



