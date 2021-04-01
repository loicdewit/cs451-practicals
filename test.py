import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

dic = [1, 2, 3, 4, 5]
i = -1
while i < 3:
    i += 1
    inp = input("type... {}".format(dic[i]))
    if inp == "back":
        print("going back...")
        i = i - 1
    else:
        print("moving on...")