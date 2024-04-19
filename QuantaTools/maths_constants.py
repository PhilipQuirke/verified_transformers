from enum import Enum


# These are tokens used in arithmetic vocab. 
class MathsToken:
    # Token indexes 0 to 9 represent digits 0 to 9
    PLUS = 10
    MINUS = 11
    EQUALS = 12
    MULT = 13
    DIV = 14
    MAX_INDEX = DIV


# These are maths behaviors: quanta we can evaluate for each node based just on that node
class MathsBehavior(Enum):

    # Minor "maths" tags related to major tag QType.MATH_ADD:
    # Addition operation "complexity" non-overlapping minor tags
    ADD_S0_TAG = "S0" # Easy. No MakeCarry1
    ADD_S1_TAG = "S1"
    ADD_S2_TAG = "S2"
    ADD_S3_TAG = "S3"
    ADD_S4_TAG = "S4"
    ADD_S5_TAG = "S5" # Hard. Multiple cascades of MakeCarry1
    ADD_PCA_TAG = "SP" # PCA is clustered aligned to the T8,T9,T10 question grouping

    # Minor "maths" tags related to major tag QType.MATH_SUB (positive-answer subtraction):
    SUB_M0_TAG = "M0"  # Answer >= 0. No BorrowOne. Easy
    SUB_M1_TAG = "M1"  # Answer > 0. Has one BorrowOne.  
    SUB_M2_TAG = "M2"  # Answer > 0. 
    SUB_M3_TAG = "M3"  # Answer > 0. 
    SUB_M4_TAG = "M4+" # Answer > 0. Has multiple cascades of BorrowOne. Hard
    SUB_PCA_TAG = "MP" # PCA is clustered aligned to the T8,T9,T10 question grouping
    
    # Minor "maths" tags related to major tag QType.MATH_NEG (negative-answer subtraction):
    NEG_N1_TAG = "N1"  # Answer < 0. Includes one BorrowOne. E.g. 100-200
    NEG_N2_TAG = "N2"  # Answer < 0. Includes two BorrowOnes. E.g. 110-200
    NEG_N3_TAG = "N3"  # Answer < 0. Has multiple cascades of BorrowOne. Hard. E.g. 111-200    
    NEG_N4_TAG = "N4+" # Answer < 0. Has multiple cascades of BorrowOne. Hard. E.g. 1111-2000   
    NEG_PCA_TAG = "NP" # PCA is clustered aligned to the T8,T9,T10 question grouping

    UNKNOWN = "Unknown"
    

# Maths algorithmic purposes: interpretations we assign to each node's calculation, partially based on its behavior
# Minor "maths" tags related to major tag QType.ALGO:
# A node may serve multiple purposes and so have more than 1 of these tags.
class MathsAlgorithm(Enum):
    ADD_A_TAG = "SA" # Addition - Base Add (Dn, D'n)
    ADD_C_TAG = "SC" # Addition - Make Carry (Dn, D'n)
    ADD_S_TAG = "SS" # Addition - Use Sum 9 (Dn, D'n)
    ADD_T_TAG = "ST" # Addition - TriCase (Dn, D'n)
  
    SUB_D_TAG = "MD" # Positive-answer Subtraction - Difference (Dn, D'n)
    SUB_B_TAG = "MB" # Positive-answer Subtraction - Borrow One (Dn, D'n)
    SUB_Z_TAG = "MZ" # Positive-answer Subtraction - Sum Zero (Dn, D'n)
    SUB_T_TAG = "MT" # Positive-answer Subtraction - TriCase (Dn, D'n)

    NEG_D_TAG = "ND" # Negative-answer Subtraction - Difference (Dn, D'n)
    NEG_B_TAG = "NB" # Negative-answer Subtraction - Borrow One (Dn, D'n)
    NEG_Z_TAG = "NZ" # Negative-answer Subtraction - Sum Zero (Dn, D'n)
    NEG_T_TAG = "NT" # Negative-answer Subtraction - TriCase (Dn, D'n)
  
    MIX_O_TAG = "OP" # Add/Sub - Attends to operation token (in the middle of the question)
    MIX_S_TAG = "SG" # Add/Sub - Attends to answer sign (+/-) token (at the start of the answer)