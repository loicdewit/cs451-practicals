import json
from dataclasses import dataclass
from typing import Dict, Any, List


matrix: List[List[float]] = []
for i in range(10):
    temp = []
    for j in range(10):
        temp.append(j)
    matrix.append(temp)

print(matrix)
