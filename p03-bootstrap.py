"""
In this lab, we'll go ahead and use the sklearn API to compare some real models over real data!
Same data as last time, for now.

Documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

We'll need to install sklearn, and numpy.
Use pip:

    # install everything from the requirements file.
    pip install -r requirements.txt
"""

#%%

# We won't be able to get past these import statments if you don't install the libraries!
# external libraries:
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# standard python
import json
from dataclasses import dataclass
from typing import Dict, Any, List

# helper functions I made
from shared import dataset_local_path, TODO

#%% load up the data
examples = []
ys = []

with open(dataset_local_path("poetry_id.jsonl")) as fp:
    for line in fp:
        info = json.loads(line)
        # Note: the data contains a whole bunch of extra stuff; we just want numeric features for now.
        keep = info["features"]
        # whether or not it's poetry is our label.
        ys.append(info["poetry"])
        # hold onto this single dictionary.
        examples.append(keep)

#%% Convert data to 'matrices'
# We did this manually in p02, but SciKit-Learn has a tool for that:

from sklearn.feature_extraction import DictVectorizer

feature_numbering = DictVectorizer(sort=True)
feature_numbering.fit(examples)
X = feature_numbering.transform(examples)

print("Features as {} matrix.".format(X.shape))
#%% Set up our ML problem:

from sklearn.model_selection import train_test_split

RANDOM_SEED = 12345678

# Numpy-arrays are more useful than python's lists.
y = np.array(ys)
# split off train/validate (tv) pieces.
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, train_size=0.75, shuffle=True, random_state=RANDOM_SEED
)
# split off train, validate from (tv) pieces.
X_train, X_vali, y_train, y_vali = train_test_split(
    X_tv, y_tv, train_size=0.66, shuffle=True, random_state=RANDOM_SEED
)

# In this lab, we'll ignore test data, for the most part.

#%% DecisionTree Parameters:
params = {
    "criterion": "gini",
    "splitter": "best",
    "max_depth": 5,
}

# train 100 different models, with different sources of randomness:
N_MODELS = 100
# sample 1 of them 100 times, to get a distribution of data for that!
N_SAMPLES = 100

seed_based_accuracies = []
for randomness in range(N_MODELS):
    f_seed = DecisionTreeClassifier(random_state=RANDOM_SEED + randomness, **params)
    f_seed.fit(X_train, y_train)
    seed_based_accuracies.append(f_seed.score(X_vali, y_vali))


bootstrap_based_accuracies = []
# single seed, bootstrap-sampling of predictions:
f_single = DecisionTreeClassifier(random_state=RANDOM_SEED, **params)
f_single.fit(X_train, y_train)
y_pred = f_single.predict(X_vali)

# do the bootstrap:
for trial in range(N_SAMPLES):
    sample_pred, sample_truth = resample(
        y_pred, y_vali, random_state=trial + RANDOM_SEED
    )
    score = accuracy_score(y_true=sample_truth, y_pred=sample_pred)
    bootstrap_based_accuracies.append(score)


boxplot_data: List[List[float]] = [seed_based_accuracies, bootstrap_based_accuracies]
plt.boxplot(boxplot_data)
plt.xticks(ticks=[1, 2], labels=["Seed-Based", "Bootstrap-Based"])
plt.xlabel("Sampling Method")
plt.ylabel("Accuracy")
plt.ylim([0.8, 1.0])
plt.show()
# if plt.show is not working, try opening the result of plt.savefig instead!
# plt.savefig("dtree-variance.png") # This doesn't work well on repl.it.

"""
Question 1: understand/compare the bounds generated between the two methods.

It seems that the mean accuracy results for the seed-based and the bootstrap-based are the same (as far as my eye can tell, which for our purposes, means they are...). However, it seems like the seed-based approach leads to more consistent accuracy levels. So, while you are more likely to have a higher accuracy value with bootstrap, you are also more likely to have a lower one - there is a lot more variability in the resulting models.

However, we can observe by increasing the scale of the y-axis that the difference in bounds is to be considered in perspective. The range makes it seems like there is a huge difference in bounds, but in reality, we are looking at a relatively small range of values,a ll above 0.875% of accuracy.
"""

# Question 2A: What happens to the variance if we do K bootstrap samples for each of M models?

f2_single = DecisionTreeClassifier(random_state=RANDOM_SEED, **params)
f2_single.fit(X_train, y_train)
y2_pred = f2_single.predict(X_vali)

K_bootstrap: List[List[float]] = []
K = 20
for i in range(K):
    K_values = []
    for boot_sample in range(N_SAMPLES):
        sample_pred, sample_truth = resample(
            y2_pred, y_vali, random_state=i * boot_sample + RANDOM_SEED
        )
        score = accuracy_score(y_true=sample_truth, y_pred=sample_pred)
        K_values.append(score)
    K_bootstrap.append(K_values)

plt.boxplot(K_bootstrap)
plt.xlabel("Depth Level 1-10 for Seed Based and Bootstrap Based Sampling Method")
plt.ylabel("Accuracy")
plt.ylim([0.7, 1.0])
plt.show()

"""
Question 2A: Based on the above experiment and the plot generated, it seems like the variance remains more or less constant.
"""

# Question 2B: modifying the plot to show ~10 depths of the decision tree

parameters = {"criterion": "gini", "splitter": "best"}
accuracies_matrix: List[List[float]] = []

for i in range(10):
    same_depth_score = []
    for randomness in range(N_MODELS):
        f_seed = DecisionTreeClassifier(
            random_state=RANDOM_SEED + randomness, max_depth=i + 1, **parameters
        )
        f_seed.fit(X_train, y_train)
        same_depth_score.append(f_seed.score(X_vali, y_vali))
    accuracies_matrix.append(same_depth_score)

    # single seed, bootstrap-sampling of predictions:
    f1_single = DecisionTreeClassifier(
        random_state=RANDOM_SEED, max_depth=i + 1, **parameters
    )
    f1_single.fit(X_train, y_train)
    y1_pred = f1_single.predict(X_vali)
    # do the bootstrap:
    for trial in range(N_SAMPLES):
        sample_pred, sample_truth = resample(
            y1_pred, y_vali, random_state=trial + RANDOM_SEED
        )
        score = accuracy_score(y_true=sample_truth, y_pred=sample_pred)
        same_depth_score.append(score)
    accuracies_matrix.append(same_depth_score)


plt.boxplot(accuracies_matrix)
plt.xlabel("Depth Level 1-10 for Seed Based and Bootstrap Based Sampling Method")
plt.ylabel("Accuracy")
plt.ylim([0.7, 1.0])
plt.show()
