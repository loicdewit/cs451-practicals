"""
In this lab, we once again have a mandatory 'python' challenge.
Then we have a more open-ended Machine Learning 'see why' challenge.

This data is the "Is Wikipedia Literary" that I pitched.
You can contribute to science or get a sense of the data here: https://label.jjfoley.me/wiki
"""

import gzip, json
from shared import dataset_local_path, TODO
from dataclasses import dataclass
from typing import Dict, List


"""
Problem 1: We have a copy of Wikipedia (I spared you the other 6 million pages).
It is separate from our labels we collected.
"""


@dataclass
class JustWikiPage:
    title: str
    wiki_id: str
    body: str


# Load our pages into this pages list.
pages: List[JustWikiPage] = []
with gzip.open(dataset_local_path("tiny-wiki.jsonl.gz"), "rt") as fp:
    for line in fp:
        entry = json.loads(line)
        pages.append(JustWikiPage(**entry))


@dataclass
class JustWikiLabel:
    wiki_id: str
    is_literary: bool


# Load our judgments/labels/truths/ys into this labels list:
labels: List[JustWikiLabel] = []
with open(dataset_local_path("tiny-wiki-labels.jsonl")) as fp:
    for line in fp:
        entry = json.loads(line)
        labels.append(
            JustWikiLabel(wiki_id=entry["wiki_id"], is_literary=entry["truth_value"])
        )


@dataclass
class JoinedWikiData:
    wiki_id: str
    is_literary: bool
    title: str
    body: str


# print(len(pages), len(labels))
# print(pages[0])
# print(labels[0])

joined_data: Dict[str, JoinedWikiData] = {}


"""
Problem 1 answer: this is definitely not a neat answer but I tried a couple libraries and did not manage to get their functions working, so I went for the brute force approach.
"""


def find_item(list_labels: List[JustWikiLabel], item: str):
    """
    Finds the element with the given idea in the list of JustWikiLabel.
    """
    for element in list_labels:
        if element.wiki_id == item:
            return element


for item in pages:
    label = find_item(labels, item.wiki_id)
    joined_data[item.wiki_id] = JoinedWikiData(
        item.wiki_id, label.is_literary, item.title, item.body
    )


############### Problem 1 ends here ###############

# Make sure it is solved correctly!
assert len(joined_data) == len(pages)
assert len(joined_data) == len(labels)
# Make sure it has *some* positive labels!
assert sum([1 for d in joined_data.values() if d.is_literary]) > 0
# Make sure it has *some* negative labels!
assert sum([1 for d in joined_data.values() if not d.is_literary]) > 0

# Construct our ML problem:
ys = []
examples = []
for wiki_data in joined_data.values():
    ys.append(wiki_data.is_literary)
    examples.append(wiki_data.body)

## We're actually going to split before converting to features now...
from sklearn.model_selection import train_test_split
import numpy as np

RANDOM_SEED = 1234

## split off train/validate (tv) pieces.
ex_tv, ex_test, y_tv, y_test = train_test_split(
    examples,
    ys,
    train_size=0.75,
    shuffle=True,
    random_state=RANDOM_SEED,
)
# split off train, validate from (tv) pieces.
ex_train, ex_vali, y_train, y_vali = train_test_split(
    ex_tv, y_tv, train_size=0.66, shuffle=True, random_state=RANDOM_SEED
)

## Convert to features, train simple model (TFIDF will be explained eventually.)
from sklearn.feature_extraction.text import TfidfVectorizer

# Only learn columns for words in the training data, to be fair.
word_to_column = TfidfVectorizer(
    strip_accents="unicode", lowercase=True, stop_words="english", max_df=0.5
)
word_to_column.fit(ex_train)

# Test words should surprise us, actually!
X_train = word_to_column.transform(ex_train)
X_vali = word_to_column.transform(ex_vali)
X_test = word_to_column.transform(ex_test)


print("Ready to Learn!")
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

models = {
    "SGDClassifier": SGDClassifier(),
    "Perceptron": Perceptron(),
    "LogisticRegression": LogisticRegression(),
    "DTree": DecisionTreeClassifier(),
}

for name, m in models.items():
    print("Printing the name and m: {} {}".format(name, m))
    m.fit(X_train, y_train)
    print("{}:".format(name))
    print("\tVali-Acc: {:.3}".format(m.score(X_vali, y_vali)))
    if hasattr(m, "decision_function"):
        scores = m.decision_function(X_vali)
    else:
        scores = m.predict_proba(X_vali)[:, 1]
    print("\tVali-AUC: {:.3}".format(roc_auc_score(y_score=scores, y_true=y_vali)))


"""
Results should be something like:

SGDClassifier:
        Vali-Acc: 0.84
        Vali-AUC: 0.879
Perceptron:
        Vali-Acc: 0.815
        Vali-AUC: 0.844
LogisticRegression:
        Vali-Acc: 0.788
        Vali-AUC: 0.88
DTree:
        Vali-Acc: 0.739
        Vali-AUC: 0.71
"""

"""
Problem 2: Explore why DecisionTrees are not beating linear models.
2.A.: I will be varying the depth to see if it improves the decision tree performance.
"""
# Experimenting with the recursion depth of the decision tree:
for i in range(20):
    f = DecisionTreeClassifier(max_depth=(i + 1) * 2)
    f.fit(X_train, y_train)
    print("\tVali-Acc: {:.3}".format(f.score(X_vali, y_vali)))
    score_dtree = m.predict_proba(X_vali)[:, 1]
    print("\tVali-AUC: {:.3}".format(roc_auc_score(y_score=score_dtree, y_true=y_vali)))
    print("\n")

"""
Changing the recursion depth for the DecisionTreeClassifier did not change its performance according to the AUC and accuracy that much - at least not enough to allow it to catch up with the other models. Thus, I am postulating that its bad performance is related to the fact that the problem at hand is linear and has a shape which does not lend itself to the "learning process" of the decision tree. I am curious to see if the random forest is able to outperform the decision tree, so I will try that next.
"""

print("Evaluating the performance of the random forest:")
random_forest = RandomForestClassifier()
params = {"criterion": "entropy"}
random_forest.fit(X_train, y_train)
print("\tVali-Acc: {:.3}".format(random_forest.score(X_vali, y_vali)))
score_forest = random_forest.predict_proba(X_vali)[:, 1]
print("\tVali-AUC: {:.3}".format(roc_auc_score(y_score=score_forest, y_true=y_vali)))


"""
Two sample runs of the code above yielded the following results for the random forest:
        Vali-Acc: 0.825
        Vali-AUC: 0.874

        Vali-Acc: 0.825
        Vali-AUC: 0.867

The random forest does therefore do better than the decision tree in this instance, as measured both by the accuracy measure and the AUC.
"""
