import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier

# new helpers:
from shared import dataset_local_path, bootstrap_accuracy, simple_boxplot, TODO

# stdlib:
from dataclasses import dataclass
import json
from typing import Dict, Any, List


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

## CONVERT TO MATRIX:

feature_numbering = DictVectorizer(sort=True)
X = feature_numbering.fit_transform(examples)

print("Features as {} matrix.".format(X.shape))


## SPLIT DATA:

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

print(X_train.shape, X_vali.shape, X_test.shape)

#%% Define & Run Experiments
@dataclass
class ExperimentResult:
    vali_acc: float
    params: Dict[str, Any]
    model: ClassifierMixin


def consider_decision_trees():
    print("Consider Decision Tree.")
    performances: List[ExperimentResult] = []

    for rnd in range(3):
        for crit in ["entropy"]:
            for d in range(1, 9):
                params = {
                    "criterion": crit,
                    "max_depth": d,
                    "random_state": rnd,
                }
                f = DecisionTreeClassifier(**params)
                f.fit(X_train, y_train)
                vali_acc = f.score(X_vali, y_vali)
                result = ExperimentResult(vali_acc, params, f)
                performances.append(result)
    return max(performances, key=lambda result: result.vali_acc)


def consider_random_forest():
    print("Consider Random Forest.")
    performances: List[ExperimentResult] = []
    # Random Forest
    for rnd in range(3):
        for crit in ["entropy"]:
            for d in range(4, 9):
                params = {
                    "criterion": crit,
                    "max_depth": d,
                    "random_state": rnd,
                }
                f = RandomForestClassifier(**params)
                f.fit(X_train, y_train)
                vali_acc = f.score(X_vali, y_vali)
                result = ExperimentResult(vali_acc, params, f)
                performances.append(result)
    return max(performances, key=lambda result: result.vali_acc)


def consider_perceptron() -> ExperimentResult:
    print("Consider Perceptron.")
    performances: List[ExperimentResult] = []
    for rnd in range(3):
        params = {
            "random_state": rnd,
            "penalty": None,
            "max_iter": 1000,
        }
        f = Perceptron(**params)
        f.fit(X_train, y_train)
        vali_acc = f.score(X_vali, y_vali)
        result = ExperimentResult(vali_acc, params, f)
        performances.append(result)

    return max(performances, key=lambda result: result.vali_acc)


def optimize_perceptron() -> ExperimentResult:
    print("Optimize Perceptron.")
    performances: List[ExperimentResult] = []
    for iter in range(1, 10):
        for pen in [
            None,
            # "I2", got an error
            # "I1", got an error
        ]:  # I tried the different penalties in the doc, but they were "not supported".
            for alpha in range(1, 10):
                for rnd in range(10):
                    params = {
                        "random_state": rnd,
                        "penalty": pen,
                        "max_iter": iter * 10,
                        "warm_start": True,
                        "alpha": 0.0001,
                    }
                    f = Perceptron(**params)
                    f.fit(X_train, y_train)
                    vali_acc = f.score(X_vali, y_vali)
                    result = ExperimentResult(vali_acc, params, f)
                    print("Result: {} \n".format(result.vali_acc))
                    performances.append(result)
    return max(performances, key=lambda result: result.vali_acc)


def consider_logistic_regression() -> ExperimentResult:
    print("Consider Logistic Regression.")
    performances: List[ExperimentResult] = []
    for rnd in range(3):
        params = {
            "random_state": rnd,
            "penalty": "l2",
            "max_iter": 100,
            "C": 1.0,
        }
        f = LogisticRegression(**params)
        f.fit(X_train, y_train)
        vali_acc = f.score(X_vali, y_vali)
        result = ExperimentResult(vali_acc, params, f)
        performances.append(result)

    return max(performances, key=lambda result: result.vali_acc)


def consider_neural_net() -> ExperimentResult:
    print("Consider Multi-Layer Perceptron.")
    performances: List[ExperimentResult] = []
    # for iter in range(1, 1000):
    #   print("Number of iteration: {}\n\n".format(iter))
    for rnd in range(3):
        # print("Rnd value {} \n\n".format(rnd))
        params = {
            "hidden_layer_sizes": (32,),
            "random_state": rnd,
            "solver": "lbfgs",
            "max_iter": 500,
            # "max_iter": iter * 1000,
            "alpha": 0.0001,
        }
        f = MLPClassifier(**params)
        f.fit(X_train, y_train)
        vali_acc = f.score(X_vali, y_vali)
        result = ExperimentResult(vali_acc, params, f)
        performances.append(result)

    return max(performances, key=lambda result: result.vali_acc)


def optimize_neural_net() -> ExperimentResult:
    print("Optimize Multi-Layer Perceptron.")
    performances: List[ExperimentResult] = []
    for iter in range(1, 10):
        for solver in ["adam", "sgd", "lbfgs"]:
            for alpha in range(1, 10):
                for rnd in range(2):
                    # print("Rnd value {} \n\n".format(rnd))
                    params = {
                        "hidden_layer_sizes": (32,),
                        "random_state": rnd,
                        "solver": solver,
                        "max_iter": iter * 500,
                        "alpha": 0.00005 * alpha,
                    }
                f = MLPClassifier(**params)
                f.fit(X_train, y_train)
                vali_acc = f.score(X_vali, y_vali)
                result = ExperimentResult(vali_acc, params, f)
                performances.append(result)

    return max(performances, key=lambda result: result.vali_acc)


# logit = consider_logistic_regression()
# perceptron = consider_perceptron()
# perceptron = optimize_perceptron()
# dtree = consider_decision_trees()
# rforest = consider_random_forest()
mlp = consider_neural_net()
# mlp = optimize_neural_net()


# print("Best Logistic Regression", logit)
# print("Best Perceptron", perceptron)
# print("Best DTree", dtree)
# print("Best RForest", rforest)
print("Best MLP", mlp)

#%% Plot Results

# Helper method to make a series of box-plots from a dictionary:
# simple_boxplot(
#     {
#         "Logistic Regression": bootstrap_accuracy(logit.model, X_vali, y_vali),
#         "Perceptron": bootstrap_accuracy(perceptron.model, X_vali, y_vali),
#         "Decision Tree": bootstrap_accuracy(dtree.model, X_vali, y_vali),
#         "RandomForest": bootstrap_accuracy(rforest.model, X_vali, y_vali),
#         "MLP/NN": bootstrap_accuracy(mlp.model, X_vali, y_vali),
#     },
#     title="Validation Accuracy",
#     xlabel="Model",
#     ylabel="Accuracy",
#     save="model-cmp.png",
# )

"""
Question 1:

'consider_decision_tress()' is a function that creates different decision tree classifiers with different parameters, assesses their performance on
the validation set, and then returns the maximum of those results as an 'ExperimentResult' object, which stores the accuracy of the model on the validation set, the parameters of the model and the model itself. 

More specifically, for 3 random seeds, the model is going have a look at all possible combinations of the parameters 'criterion' and 'max_depth', where max_depth runs from 1 to 8, and criterion is for now a single option, but we could easily add 'Gini' to the list to have two.

In short, the function returns the best model of the combinations of random_seed (3) x criterion (1) x max_depth (8).

"""

"""
Question 2:

The warning messages come from the function developing the neural network.

Note: I observed that the warning was only happening when the parameter 'random_state' was set to rnd = 2. Thus, I stopped considering the other two, and I ran my for loop (see function 'consider_neural_net()') above for a large range of max_iterations. I ran the function for up to 100 000 iterations, but still it was not enough to make the warning go away, so I stopped there, as my computer was becoming very slow and heating quity a bit.

"""

"""
Question 3:

I chose to improve the perceptron! The original best was (vali_acc=0.9038189533239038, params={'random_state': 0, 'penalty': None, 'max_iter': 1000}, model=Perceptron(), alpha = 0.0001)

I tried the different penalty parameters in the documentation, but they were not supported. 

I could not find a way to optimize the perceptron. Changing the parameters max_iter and random seed only gave worse or as good models as the one above.

Saddened, I tried to optimize the neural network!

The original best was: 

Best MLP ExperimentResult(vali_acc=0.925035360678925, params={'hidden_layer_sizes': (32,), 'random_state': 1, 'solver': 'lbfgs', 'max_iter': 500, 'alpha': 0.0001}, model=MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=1, solver='lbfgs'))

 I got a small improvement with the following, obtained with the setup of function 'optimize_neural_net()'. 

Best MLP ExperimentResult(vali_acc=0.9306930693069307, params={'hidden_layer_sizes': (32,), 'random_state': 1, 'solver': 'lbfgs', 'max_iter': 500, 'alpha': 0.00025}, model=MLPClassifier(alpha=0.00025, hidden_layer_sizes=(32,), max_iter=500, random_state=1, solver='lbfgs')).

"""
