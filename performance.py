import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

# Predict on the data
pred  = model.predict(x)

# ROC calculations
steps = 10000
lin = np.linspace(0, 1, steps)

z = 1-y

TruePosRate = []
FalsePosRate = []
TrueNegRate = []
FalseNegRate = []
sig = []

for i in lin:
 TP = np.sum((pred > i) & y.astype(int))
 FP = np.sum((pred > i) & z.astype(int))
 TN = np.sum((pred < i) & z.astype(int))
 FN = np.sum((pred < i) & y.astype(int))
 sig += TP/np.sqrt(FP)
 try: TruePosRate += [TP/(TP+FN)]
 except: pass
 try: FalsePosRate += [FP/(FP+TN)]
 except: pass
 try: TrueNegRate += [TN/(TN+FP)]
 except: pass
 try: FalseNegRate += [FN/(FN+TP)]
 except: pass

TruePosRate = np.array(TruePosRate)
FalsePosRate = np.array(FalsePosRate)
TrueNegRate = np.array(TrueNegRate)
FalseNegRate = np.array(FalseNegRate)

sign = np.array(sign)
sign_idx = np.argmax(sign)
cut_value = lin[sign_idx]

plt.figure()
plt.plot(lin, sign, label='Sign')
plt.xlabel("Cut value")
plt.ylabel("Significance")
plt.axvline(x = cut_value, ymin = 0, ymax = sign[sign_idx], color = 'black', linestyle = 'dashed’, label = 'Cutvalue')
plt.yscale('log')
plt.legend(loc='lower center', frameon=False)
plt.show()

plt.figure()
plt.plot(FalsePosRate,TruePosRate,label='CNN')
plt.plot(lin, lin, linestyle = '--', color = 'black')
plt.axvline(x = FalsePosRate[sign_idx], ymin = 0, ymax = TruePosRate[sign_idx], color = 'black', linestyle = 'dashed’)
plt.axhline(y = TruePosRate[sign_idx], xmin = 0, xmax = FalsePosRate[sign_idx], color = 'black', linestyle = 'dashed’)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend(frameon=False)
plt.show()

plt.figure()
(counts, edges, plot) = plt.hist(pred, bins=100, range=(0,1), color=None, histtype='step')
plt.axvline(x = cut_value, color = 'black', linestyle = 'dashed’, label = 'Cutvalue')
plt.xlabel("Sig_Prob")
plt.ylabel("Number of CT scans")
plt.yscale('log')
plt.legend(loc='upper center', frameon=False)
plt.show()