# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# Adapted and modified to be used as a function by Waïss Azizian
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, FuncNorm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.metrics import f1_score

h = .02  # step size in the mesh

param = 10
def forward_norm(arr):
	return 1/(1+np.exp(-param * (arr - 0.5)))
def backward_norm(arr):
	return (1/param) * np.log(arr/(1 - arr)) +0.5
normalizer = FuncNorm((forward_norm, backward_norm), vmin=0., vmax=1.)

def plot_classifier_comparison(names, classifiers, datasets, levels=10, save=None):
	assert len(names) == len(classifiers)
	figure = plt.figure(figsize=(32, 7.5)) #figsize=(32, 15)
	i = 1
	# iterate over datasets
	cols = len(classifiers) + 2
	for ds_cnt, ds in enumerate(datasets):
		# preprocess dataset, split into training and test part
		dataset_train, dataset_test = ds
		X_train, y_train = dataset_train
		X_test, y_test = dataset_test
		y_train = np.sign( y_train-0.5)
		y_test = np.sign( y_test-0.5)
		x_min = min(X_train[:, 0].min(), X_test[:, 0].min()) - .5
		x_max = max(X_train[:, 0].max(), X_test[:, 0].max()) + .5
		y_min = min(X_train[:, 1].min(), X_test[:, 1].min()) - .5
		y_max = max(X_train[:, 1].max(), X_test[:, 1].max()) + .5
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
							 np.arange(y_min, y_max, h))

		# just plot the dataset first
		cm = plt.cm.RdBu
		cm_bright = ListedColormap(['#FF0000', '#0000FF'])

		ax = plt.subplot(len(datasets), cols, i)
		if ds_cnt == 0:
			ax.set_title("Training data",size=40)
		# Plot the training points
		ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
				   edgecolors='k')
		ax.set_xlim(xx.min(), xx.max())
		ax.set_ylim(yy.min(), yy.max())
		ax.set_xticks(())
		ax.set_yticks(())
		i += 1

		ax = plt.subplot(len(datasets), cols, i)
		if ds_cnt == 0:
			ax.set_title("Testing data",size=40)
		# Plot the testing points
		ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
				   edgecolors='k')
		ax.set_xlim(xx.min(), xx.max())
		ax.set_ylim(yy.min(), yy.max())
		ax.set_xticks(())
		ax.set_yticks(())
		i += 1

		# iterate over classifiers
		for name, clf in zip(names, classifiers):
			ax = plt.subplot(len(datasets), cols, i)
			clf.fit(X_train, y_train)
			score = clf.score(X_test, y_test)
			#f1_sc = f1_score(y_test, clf.predict(X_test)) 

			# Plot the decision boundary. For that, we will assign a color to each
			# point in the mesh [x_min, x_max]x[y_min, y_max].
			#if hasattr(clf, "decision_function"):
			#	Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
			#else:
			Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

			# Put the result into a color plot
			Z = Z.reshape(xx.shape)
			cs = ax.contourf(xx, yy, Z, cmap=cm, alpha=.8, levels=levels, norm=normalizer)
			#cs = ax.contour(xx, yy, Z, cmap=cm, alpha=.8, levels=levels)
			levels = cs.levels
			#plt.colorbar(cs)

			# Plot the training points
			ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
					   edgecolors='k')
			# Plot the testing points
			ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
					   edgecolors='k', alpha=0.6)

			ax.set_xlim(xx.min(), xx.max())
			ax.set_ylim(yy.min(), yy.max())
			ax.set_xticks(())
			ax.set_yticks(())
			if ds_cnt == 0:
				ax.set_title(name,size=40)
			ax.text(xx.max() - .3, yy.min() + .3, f"Acc. {int(score*100)}%",#('%.2f' % score).lstrip('0'),
					size=50, horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.9))
			i += 1
		# Less than ideal but will do thx to trim
		#ax = plt.subplot(len(datasets), cols, i)
		#plt.colorbar(cs, ax=ax, location='left')
		i += 1

	plt.tight_layout()
	if save is not None:
		plt.savefig(save)
	#plt.show()




