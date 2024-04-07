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
    # Minor "maths" tags related to major tag QType.PCA:
    # PCA says the node outputs is interpretable aligned to the T8,T9,T10 questions, giving 2 or 3 distinct output clusters
    PCA_ADD_TAG = "PA"
    PCA_SUB_TAG = "PS"

    # Minor "maths" tags related to major tag QType.MATH_ADD:
    # Addition operation "complexity" non-overlapping minor tags
    ADD_S0_TAG = "S0" # Easy. No MakeCarry1
    ADD_S1_TAG = "S1"
    ADD_S2_TAG = "S2"
    ADD_S3_TAG = "S3"
    ADD_S4_TAG = "S4"
    ADD_S5_TAG = "S5" # Hard. Multiple cascades of MakeCarry1

    # Minor "maths" tags related to major tag QType.MATH_SUB:
    # Subtraction operation "complexity" non-overlapping minor tags
    SUB_M0_TAG = "M0"  # Answer >= 0. No BorrowOne. Easy
    SUB_M1_TAG = "M1"  # Answer > 0. Has one BorrowOne.  
    SUB_M2_TAG = "M2"  # Answer > 0. 
    SUB_M3_TAG = "M3"  # Answer > 0. 
    SUB_M4_TAG = "M4+" # Answer > 0. Has multiple cascades of BorrowOne. Hard
    # Minor "maths" tags related to major tag QType.MATH_NEG:
    SUB_N1_TAG = "N1"  # Answer < 0. Includes one BorrowOne. E.g. 100-200
    SUB_N2_TAG = "N2"  # Answer < 0. Includes two BorrowOnes. E.g. 110-200
    SUB_N3_TAG = "N3"  # Answer < 0. Has multiple cascades of BorrowOne. Hard. E.g. 111-200    
    SUB_N4_TAG = "N4+" # Answer < 0. Has multiple cascades of BorrowOne. Hard. E.g. 1111-2000   

    UNKNOWN = "Unknown"
    

# These are maths algorthmic purposes: interpretations we assign to each node, partially based on its behavior
# Minor "maths" tags related to major tag QType.ALGO:
# A node may serve multiple purposes and so have more than 1 of these tags.
class MathsAlgorithm(Enum):
    ADD_BA_TAG = "BA" # Addition - Base Add (Dn, D'n)
    ADD_MC_TAG = "MC" # Addition - Make Carry (Dn, D'n)
    ADD_US_TAG = "US" # Addition - Use Sum 9 (Dn, D'n)
    ADD_TC_TAG = "TC" # Addition - TriCase (Dn, D'n)
  
    SUB_BS_TAG = "BS" # Subtraction - Base Sub (Dn, D'n)
    SUB_BO_TAG = "BO" # Subtraction - Borrow One (Dn, D'n)
    SUB_SZ_TAG = "SZ" # Subtraction - Sum Zero (Dn, D'n)
    SUB_NG_TAG = "NG" # Subtraction - Answer is negative (that is D < D')
  
    MIX_OP_TAG = "OP" # Add/Sub - Attends to operation token (in the middle of the question)
    MIX_SG_TAG = "SG" # Add/Sub - Attends to answer sign (+/-) token (at the start of the answer)