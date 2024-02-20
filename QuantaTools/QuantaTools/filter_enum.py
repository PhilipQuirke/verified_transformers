import sys
from enum import Enum

class Filter(Enum):
    MUST = 1  # The specified quanta value MUST exist in the object
    NOT = 2  # The specified quanta value must NOT exist in the object
    CONTAINS = 3 # The specified quanta value can equal the object value or a subset of the object value
    MAY = 4  # The specified quanta value MAY exist (Indefinite. Used in a hypothesis)

    
    def __eq__(self, other):
        """Necessary because of extremely frustrating error that arises with load_ext autoreload (because this uses importlib under the hood: https://stackoverflow.com/questions/66458864/enum-comparison-become-false-after-reloading-module)"""

        assert isinstance(other, Filter)
        return self.value == other.value
