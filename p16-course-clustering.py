import pandas as pd
from sklearn import cluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plot
from sklearn.cluster import KMeans, AgglomerativeClustering


# Classify course a number based on its level
def course_to_level(num: int) -> int:
    if num < 200:
        return 1
    elif num < 300:
        return 2
    elif num < 400:
        return 3
    elif num < 500:
        return 4
    elif num < 600:
        return 5
    elif num >= 1000:
        return 1
    else:
        return 5


# Read in the data
df = pd.read_json("data/midd_cs_courses.jsonl", lines=True)

print(df[["number", "title"]])

vectorizer = TfidfVectorizer()
# Tfidf Vectorize the description feature
X = vectorizer.fit_transform(df.description).toarray()
# Extract the number feature from the examples
numbers = df.number.array.to_numpy()
# Convert level feature to
levels = [course_to_level(n) for n in df.number.to_list()]

## TODO: improve this visualization of CS courses.
#

## IDEAS: compare PCA to TSNE
# PCA doesn't have a perplexity parameter.
# What does TSNE do better on this dataset?
"""
PCA clusters the majority of the data in the bottom left corner of the chart, with a second large cluster to the bottom right, and then a few outliers in the top right of the charet. The TSNE spreads the data more, and seems to cluster the data more sensibly, as shown by labelling the data with the k-means labels or their course level label.
"""
perplexity = 15
pca = True
if pca:
    viz = PCA(random_state=42)
    fill = "PCA(Courses)"
else:
    viz = TSNE(perplexity=perplexity, random_state=42)
    fill = "T-SNE(Courses), perplexity={}".format(perplexity)

V = viz.fit_transform(X)

## IDEAS: kmeans
# Create a kmeans clustering of the courses.
# Then apply colors based on the kmeans-prediction to the below t-sne graph.
num_kclusters = 5
k_means = KMeans(n_clusters=num_kclusters, random_state=42)
k_means.fit_transform(X)

# Right now, let's assign colors to our class-nodes based on their number.
"""
Swapped to put the k_means labels instead.
"""
color_values = k_means.labels_

plot.title("{}".format(fill))
plot.scatter(V[:, 0], V[:, 1], alpha=1, s=10, c=color_values, cmap="turbo")

# Annotate the scattered points with their course number.
for i in range(len(numbers)):
    course_num = str(numbers[i])
    x = V[i, 0]
    y = V[i, 1]
    plot.annotate(course_num, (x, y))

plot.savefig("graphs/p16-tsne-courses-p{}.png".format(perplexity))
plot.show()
