# Decision Trees: Feature Splits

#%%
# Python typing introduced in 3.5: https://docs.python.org/3/library/typing.html
from typing import List, Optional

# As of Python 3.7, this exists! https://www.python.org/dev/peps/pep-0557/
from dataclasses import dataclass

# My python file (very limited for now, but we will build up shared functions)
from shared import TODO

#%%
# Let's define a really simple class with two fields:
@dataclass
class DataPoint:
    temperature: float
    frozen: bool

    def secret_answer(self) -> bool:
        return self.temperature <= 32

    def clone(self) -> "DataPoint":
        return DataPoint(self.temperature, self.frozen)


# Fahrenheit, sorry.
data = [
    # vermont temperatures; frozen=True
    DataPoint(0, True),
    DataPoint(-2, True),
    DataPoint(10, True),
    DataPoint(11, True),
    DataPoint(6, True),
    DataPoint(28, True),
    DataPoint(31, True),
    # warm temperatures; frozen=False
    DataPoint(33, False),
    DataPoint(45, False),
    DataPoint(76, False),
    DataPoint(60, False),
    DataPoint(34, False),
    DataPoint(98.6, False),
]


def is_water_frozen(temperature: float) -> bool:
    """
    This is how we **should** implement it.
    """
    return temperature <= 32


# Make sure the data I invented is actually correct...
for d in data:
    assert d.frozen == is_water_frozen(d.temperature)


def find_candidate_splits(datapoints: List[DataPoint]) -> List[float]:
    """
    Iterative method to find the split points.
    """
    midpoints = []
    sorted_data = sorted(datapoints, key=lambda datapoint: datapoint.temperature)

    for d in range(len(sorted_data)):
        if d != len(sorted_data) - 1:
            point1 = sorted_data[d].temperature
            point2 = sorted_data[d + 1].temperature
            midpoint = ((point2 - point1) / 2) + (point1)
            midpoints.append(midpoint)

    return midpoints


def gini_impurity(points: List[DataPoint]) -> float:
    """
    The standard version of gini impurity sums over the classes:
    """

    p_ice = sum(1 for x in points if x.frozen) / len(points)
    p_water = 1.0 - p_ice
    return p_ice * (1 - p_ice) + p_water * (1 - p_water)
    # for binary gini-impurity (just two classes) we can simplify, because 1 - p_ice == p_water, etc.
    # p_ice * p_water + p_water * p_ice
    # 2 * p_ice * p_water
    # not really a huge difference.


def impurity_of_split(points: List[DataPoint], split: float) -> float:
    """
    Iterative method to split the data points into two arrays based on the split point provided
    and return the gini impurity measure for that split point.
    """
    smaller = []
    bigger = []

    sorted_data = sorted(points, key=lambda datapoint: datapoint.temperature)
    index = 0
    for d in range(len(sorted_data)):
        if d != len(sorted_data) - 1:
            if (
                sorted_data[d].temperature < split
                and sorted_data[d + 1].temperature > split
            ):
                index = d + 1
                break

    smaller = sorted_data[:index]
    bigger = sorted_data[index:]

    return gini_impurity(smaller) + gini_impurity(bigger)


def impurity_of_split_rec(points: List[DataPoint], split: float) -> float:
    """
    Recursive method wrapper to split the data points into two arrays based on the split point provided and return the gini impurity measure for that split point.
    """
    print("Printing split: {}".format(split))

    smaller = []
    bigger = []

    sorted_data = sorted(points, key=lambda datapoint: datapoint.temperature)

    splitpoint = __impurity_of_split_rec(sorted_data, split, 0, len(sorted_data))

    print("Printing splitpoint {}".format(splitpoint))

    smaller = sorted_data[:splitpoint]
    bigger = sorted_data[splitpoint:]

    return gini_impurity(smaller) + gini_impurity(bigger)


def __impurity_of_split_rec(
    points: List[DataPoint], split: float, left: int, right: int
) -> Optional[int]:
    """
    "Private" recursive method called by impurity_of_split_rec to find the actual split.
    It is basically a binary search procedure.
    """
    assert left >= 0
    assert right >= 0

    mid = (right - left) // 2 + left

    if left >= right:
        return None
    else:
        temp1 = points[mid - 1].temperature
        temp2 = points[mid].temperature

        if temp1 < split and temp2 > split:
            print("Printing midpoint: {}".format(mid))
            return mid

        elif temp1 < split and temp2 < split:
            return __impurity_of_split_rec(points, split, mid, right)

        else:
            return __impurity_of_split_rec(points, split, left, mid)


if __name__ == "__main__":
    print("Initial Impurity: ", gini_impurity(data))
    print("Impurity of first-six (all True): ", gini_impurity(data[:6]))
    print("")
    for split in find_candidate_splits(data):
        score = impurity_of_split_rec(data, split)
        print("splitting at {} gives us impurity {}".format(split, score))
        if score == 0.0:
            break
