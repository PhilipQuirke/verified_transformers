from enum import Enum


class QuantaFilter(Enum):
    MUST = "must" # The specified quanta value MUST exist in the object
    NOT = "not"  # The specified quanta value must NOT exist in the object
    CONTAINS = "contains" # The specified quanta value can equal the object value or a subset of the object value
    MAY = "may"  # The specified quanta value MAY exist (Indefinite. Used in a hypothesis)
    MUST_BY = "must-by"  # The specified quanta value MUST exist in the object at or before the specified position       
