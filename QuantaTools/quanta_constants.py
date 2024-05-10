from enum import Enum

# Set of distinct conditions using in filtering. Each filter contains zero or one of these conditions
class QCondition(Enum):
    MUST = "must" # The specified quanta value MUST exist in the object
    NOT = "not"  # The specified quanta value must NOT exist in the object
    CONTAINS = "contains" # The specified quanta value can equal the object value or a subset of the object value
    MAY = "may"  # The specified quanta value MAY exist (Indefinite. Used in a hypothesis)
    MAX = "max"  # The specified quanta value MUST exist at or before the specified position       
    MIN = "min"  # The specified quanta value MUST exist at or after the specified position       


# Set of (independent) quanta "types". Each node tag contains exactly one of these quanta types
class QType(Enum):
    # GENERIC QUANTA TYPES:
    # Some quanta types apply to all models

    # (Input or answer) token position tag
    POSN = "Posn"
  
    # What % of questions failed when we ablated a specific node. A low percentage indicates a less common use case
    FAIL = "Fail%"
  
    # What input tokens (e.g. "D'3") are attended to by a specific attention head.
    ATTN = "Attn"
  
    # What answer digits (e.g. "A543") were impacted when we ablated a specific node.
    IMPACT = "Impact"
  
    # What algorithmic purpose is this node serving?
    ALGO = "Algo"

    # What are the dependencies between nodes? (future)
    ACDC = "ACDC"
    
    UNKNOWN = "Unknown"

    # MODEL-SPECIFIC QUANTA TYPES:
    # These quanta types apply to specific models predicting specific types of questions
    # They are defined here to 1) avoid naming conflicts and 2) act as an 'index' of implemented quanta
    
    # Types of mathematical questions 
    MATH = "Math" 
    MATH_ADD = "Math.Add" 
    MATH_SUB = "Math.Sub" # Subtraction with a positive answer
    MATH_NEG = "Math.Neg" # Subtraction with a negative answer
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


# Related to QType.FAIL:
NO_IMPACT_TAG = "(none)"
# Number of shades used in quanta maps
FAIL_SHADES = 10 

# Related to QType.ATTN:
# For each node, we store at most 5 input attention facts (as tags)
MAX_ATTN_TAGS = 5
# When graphing, we only show input tokens with > 10% of the node's attention
MIN_ATTN_PERC = 10
# When graphing, if the two top percent attentions are within 5% (e.g. 'Attn:P3=52', 'Attn:P10=48')
# show the top two tokens in alphabetical order. For a task split across two adjacent cells, this helps the 
# two cells to generate the same attention text, and so for the cells to combine in graphs.
ATTN_ORDER_DIFF = 5
# When graphing, if the two top percent attentions are each over 40% (e.g. 'Attn:P5=56', 'Attn:P12=43')
# show the top two tokens in alphabetical order. For a task split across two adjacent cells, this helps the 
# two cells to generate the same attention text, and so for the cells to combine in graphs.
ATTN_ORDER_MIN = 40


# Number of shades used in quanta maps
ATTN_SHADES = 10 

# Related to QType.IMPACT:
# Used when no answer digits were impacted by the intervention
NO_IMPACT_TAG = "(none)"

# Related to QType.ALGO:
# Number of shades used in quanta maps
ALGO_SHADES = 2

# Related to QType.MATH_*:
# Number of shades used in quanta maps for mathematical questions
MATH_ADD_SHADES = 6
MATH_SUB_SHADES = 5
