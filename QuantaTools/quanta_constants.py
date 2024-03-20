from enum import Enum


class QCondition(Enum):
    MUST = "must" # The specified quanta value MUST exist in the object
    NOT = "not"  # The specified quanta value must NOT exist in the object
    CONTAINS = "contains" # The specified quanta value can equal the object value or a subset of the object value
    MAY = "may"  # The specified quanta value MAY exist (Indefinite. Used in a hypothesis)
    MUST_BY = "must-by"  # The specified quanta value MUST exist in the object at or before the specified position       


class QType:
    # Token position tag
    POSITION = "Position"
  
    # What % of questions failed when we ablated a specific node. A low percentage indicates a less common use case
    FAIL = "Fail%"
  
    # What input tokens (e.g. "D'3") are attended to by a specific attention head.
    ATTENTION = "Attn"
  
    # What answer digits (e.g. "A543") were impacted when we ablated a specific node.
    IMPACT = "Impact"
  
    # What does Principal Component Analysis say about the node?
    PCA = "PCA"
  
    # What algorithmic purpose is this node serving?
    ALGO = "Algorithm"


    # Types of mathematical operations (questions)
    MATH_ADD = "Math.Add" 
    MATH_SUB = "Math.Sub"
    MATH_MUL = "Math.Mul"
  
    # Types of logic questions (future)
    LOGIC_1 = "Logic.1" 
    LOGIC_2 = "Logic.2"
    LOGIC_3 = "Logic.3"
    LOGIC_4 = "Logic.4"
    
    # Types of questions (future)
    SPARE_1 = "Spare.1"
    SPARE_2 = "Spare.2"
    SPARE_3 = "Spare.3"
    SPARE_4 = "Spare.4"

  

# Related to QType.ATTENTION:
# For each node, we store at most 5 input attention facts (as tags)
MAX_ATTENTION_TAGS = 5
# When graphing, we only show input tokens with > 10% of the node's attention
MIN_ATTENTION_PERC = 10


# Related to QType.IMPACT:
# No answer digits were impacted by the intervention
NO_IMPACT_TAG = "(none)"
