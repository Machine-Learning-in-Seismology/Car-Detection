import pickle5 as pickle
from pickle import dump
import numpy as np
import pandas as pd
import os, glob, random, time, sys
import matplotlib.pyplot as plt
from model_fft import build_nn, get_early_stop
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from datetime import datetime
from obspy import Trace
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

DEBUG_THRESHOLD = False


def plot_signal(s,f):
	fig, axs = plt.subplots(nrows=2,ncols=3)
	axs[0,0].plot(s[:,0], color="tab:orange")
	axs[1,0].plot(f[:,0], color="tab:orange")
	axs[0,1].plot(s[:,1], color="tab:blue")
	axs[1,1].plot(f[:,1], color="tab:blue")
	axs[0,2].plot(s[:,2], color="tab:green")
	axs[1,2].plot(f[:,2], color="tab:green")
	fig.tight_layout()
	plt.show()

EPOCHS = [20] # 20

dlen, norm = int(sys.argv[1]), sys.argv[2]
dlen_npts = dlen * 100

print("Loading data...")
MASTER_DIR = "../DBs/" + str(dlen) + "s"
# Waveform
with open(os.path.join(MASTER_DIR, f"earthquakes_ps7sec_ms2.pkl"), "rb") as f:
	eq_wf = pickle.load(f, encoding='latin1')
with open(os.path.join(MASTER_DIR, f"cars_ms2.pkl"), "rb") as f:
	car_wf = pickle.load(f, encoding='latin1')
with open(os.path.join(MASTER_DIR, f"instance_noises_ms2.pkl"), "rb") as f: #_cars
	noise_wf = pickle.load(f, encoding='latin1')
# FFT
with open(os.path.join(MASTER_DIR, f"earthquakes_fft_ps7sec_ms2.pkl"), "rb") as f:
	eq_fft = pickle.load(f, encoding='latin1')
with open(os.path.join(MASTER_DIR, f"cars_fft_ms2.pkl"), "rb") as f:
	car_fft = pickle.load(f, encoding='latin1')
with open(os.path.join(MASTER_DIR, f"instance_noises_fft_ms2.pkl"), "rb") as f: #_cars
	noise_fft = pickle.load(f, encoding='latin1')

# Labels
y_noise_wf = [0] * len(noise_wf)
y_eq_wf = [1] * (len(eq_wf))
y_car_wf = [2] * (len(car_wf))
y_wf = np.array(y_noise_wf + y_eq_wf + y_car_wf)
# Data
# Waveform
noise_wf = np.array(noise_wf)
eq_wf = np.array(eq_wf)
car_wf = np.array(car_wf)
print(noise_wf.shape)
print(eq_wf.shape)
print(car_wf.shape)
x_wf = np.append(noise_wf,eq_wf,axis=0)
x_wf = np.append(x_wf,car_wf,axis=0)
# FFT
noise_fft = np.array(noise_fft)
eq_fft = np.array(eq_fft)
car_fft = np.array(car_fft)
print(noise_fft.shape)
print(eq_fft.shape)
print(car_fft.shape)
x_fft = np.append(noise_fft,eq_fft,axis=0)
x_fft = np.append(x_fft,car_fft,axis=0)

X_wf = np.zeros([len(x_wf), dlen_npts, 3])
X_wf = np.zeros([len(x_wf), dlen_npts, 3])
for i, z in enumerate(x_wf):
	# print(i)
	if len(z.shape) == 1:
		len_min = np.min([len(z[0]),len(z[1]),len(z[2])])
		z[0] = z[0][:len_min]
		z[1] = z[1][:len_min]
		z[2] = z[2][:len_min]

		z2 = np.array([z[0],z[1],z[2]])
		z = z2
	z1 = np.zeros((3,dlen_npts))
	if norm == 'y':
		z1[0,:] = z[0,:]/max(abs(z[0,:]))
		z1[1,:] = z[1,:]/max(abs(z[1,:]))
		z1[2,:] = z[2,:]/max(abs(z[2,:]))
	else:
		z1[0,:] = z[0,:]
		z1[1,:] = z[1,:]
		z1[2,:] = z[2,:]
	X_wf[i, :, :z1.shape[1]] = z1.T

if dlen == 10:
	X_spec = np.zeros([len(x_fft), int(dlen_npts/2), 3])
elif dlen == 20:
	X_spec = np.zeros([len(x_fft), int(dlen_npts/4), 3])

for i, z in enumerate(x_fft):
	if dlen == 10:
		z1 = np.zeros((3,int(dlen_npts/2)))
	elif dlen == 20:
		z1 = np.zeros((3,int(dlen_npts/4)))
	z1[0,:] = z[0,:]/max(abs(z[0,:]))
	z1[1,:] = z[1,:]/max(abs(z[1,:]))
	z1[2,:] = z[2,:]/max(abs(z[2,:]))
	X_spec[i, :, :z1.shape[1]] = z1.T


results = pd.DataFrame(columns=['seed', 'fold', 'acc', 'cat_crossentro', 'Precision', 'Recall', 'F1_score', 'Cohens_kappa', 'ROC_AUC', 'Confusion_Matrix'])
timestr = time.strftime("%Y%m%d-%H%M%S")
print(timestr)

y_wf = to_categorical(y_wf)
np.random.seed(42)
shuffled_indices = np.random.permutation(len(y_wf))
X_wf = X_wf[shuffled_indices]
X_spec = X_spec[shuffled_indices]
y_wf = y_wf[shuffled_indices]

# Build the model
nn = build_nn(X_wf.shape[1:],X_spec.shape[1:])

# Kfold
kf = KFold(n_splits=4,shuffle=False)

fold = 0
for train_index, test_index in kf.split(X_wf):
	# WF
	X_wf_train, X_wf_test = X_wf[train_index], X_wf[test_index]
	y_wf_train, y_wf_test = y_wf[train_index], y_wf[test_index]
	# FFT
	X_fft_train, X_fft_test = X_spec[train_index], X_spec[test_index]
	for seed in range(5):
		for epoch in EPOCHS:

			# Give a name to model
			f_name = 'Models/model_fft_' + str(dlen) + "s_"  + 'norm_' + norm + '_' + timestr  + '.h5'
			log_name = "logs/log_fft" + str(dlen) + "s_" + timestr + ".csv"
			history_logger=tf.keras.callbacks.CSVLogger(log_name, separator=",", append=True)
			# Fit the model
			mc = ModelCheckpoint(filepath=f_name, monitor='val_loss', save_best_only=True)

			test_score = nn.fit(x=[X_wf_train,X_fft_train], y=y_wf_train,
				   batch_size=100, epochs=epoch,
				   validation_split=0.25,
				   callbacks=[get_early_stop(),mc,history_logger])
			
			score = nn.evaluate([X_wf_test, X_fft_test], [y_wf_test, y_wf_test])
			''' Scores:
			Accuracy
			CategoricalCrossentropy
			'''

			y_pred = nn.predict([X_wf_test, X_fft_test])
			# accuracy: (tp + tn) / (p + n)
			accuracy = accuracy_score(y_wf_test.argmax(axis=1), y_pred.argmax(axis=1))
			# precision tp / (tp + fp)
			precision = precision_score(y_wf_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
			# recall: tp / (tp + fn)
			recall = recall_score(y_wf_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
			# f1: 2 tp / (2 tp + fp + fn)
			f1 = f1_score(y_wf_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
			# kappa
			kappa = cohen_kappa_score(y_wf_test.argmax(axis=1), y_pred.argmax(axis=1))
			# ROC AUC
			auc = roc_auc_score(y_wf_test, y_pred,multi_class='ovr')
			# confusion matrix
			matrix = confusion_matrix(y_wf_test.argmax(axis=1), y_pred.argmax(axis=1))

			results.loc[len(results)] = [seed, fold, score[1], score[2], precision, recall, f1, kappa, auc, matrix]
			results.to_csv("results/model_fft" + str(dlen) + "s_results" + '_norm_' + norm + '_' + timestr + ".csv", sep="\t", encoding='utf-8')
	fold += 1
