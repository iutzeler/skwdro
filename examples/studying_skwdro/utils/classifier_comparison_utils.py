# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# Adapted and modified to be used as a function by Waïss Azizian
# Re-adapted by Florian Vincent
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, FuncNorm
from sklearn.metrics import f1_score, log_loss
from tqdm import tqdm

h = .02  # step size in the mesh

param = 10
def forward_norm(arr):
	return 1/(1+np.exp(-param * (arr - 0.5)))
def backward_norm(arr):
	return (1/param) * np.log(arr/(1 - arr)) +0.5
normalizer = FuncNorm((forward_norm, backward_norm), vmin=0., vmax=1.)

def plot_classifier_comparison(names, classifiers, datasets, levels=10):
	assert len(names) == len(classifiers)
	figure = plt.figure(figsize=(32, 15))
	i = 1
	# iterate over datasets
	cols = len(classifiers) + 2
	for ds_cnt, ds in enumerate(datasets):
		# preprocess dataset, split into training and test part
		dataset_train, dataset_test = ds
		X_train, y_train = dataset_train
		X_test, y_test = dataset_test
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
			ax.set_title("Training data")
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
			ax.set_title("Testing data")
		# Plot the testing points
		ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
				   edgecolors='k')
		ax.set_xlim(xx.min(), xx.max())
		ax.set_ylim(yy.min(), yy.max())
		ax.set_xticks(())
		ax.set_yticks(())
		i += 1


		# iterate over classifiers
		iterator = tqdm(
		    zip(names, classifiers),
		    total=len(classifiers),
		    leave=True
		)
		for name, clf in iterator:
			ax = plt.subplot(len(datasets), cols, i)
			clf.fit(X_train, y_train)
			score = clf.score(X_test, y_test)
			f1_sc = f1_score(y_test, clf.predict(X_test));
			reference_erm_loss = log_loss(
				y_test,
				classifiers[0].predict_proba_2Class(X_test)
			)
			robloss = log_loss(
			    y_test,
			    clf.predict_proba_2Class(X_test)
			)
			test_loss_gap = (robloss - reference_erm_loss) / np.abs(reference_erm_loss)
			iterator.set_postfix_str(f"""
                {robloss:.3f}/{reference_erm_loss:.3f}
            """)

			# Plot the decision boundary.
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
				ax.set_title(name)
			ax.text(
                xx.max() - .3,
                yy.min() + .3,
                f"Test Acc. {int(score*100)}%\nF1 {int(f1_sc*100)}%\nLoss-gap {int(test_loss_gap*100)}%",
				size=12,
                horizontalalignment='right',
                bbox=dict(facecolor='wheat', alpha=0.5)
            )
			#print(f"Dataset {ds_cnt}, Classifier {name}: Test Acc. {int(score*100)}%, F1 {int(f1_sc*100)}%")
			i += 1


	plt.tight_layout()
	plt.show()



