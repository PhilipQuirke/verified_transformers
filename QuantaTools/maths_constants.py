class MathsTokens:
    # Tokens used in arithmetic vocab. (Token indexes 0 to 9 represent digits 0 to 9)
    PLUS = 10
    MINUS = 11
    EQUALS = 12
    MULT = 13
    DIV = 14
    MAX_INDEX = DIV


# These are maths behaviors: quanta we can evaluate for each node based just on that node
class MathsBehavior:
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
    SUB_S0_TAG = "M0" # Easy. No BorrowOne. Answer is positive
    SUB_S1_TAG = "M1"
    SUB_S2_TAG = "M2"
    SUB_S3_TAG = "M3" # Hard. Multiple cascades of BorrowOne. Answer is positive
    SUB_NG_TAG = "NG" # Hard. Subtraction question has negative answer


# These are maths algorthmic purposes: interpretations we assign to each node, partially based on its behavior
# Minor "maths" tags related to major tag QType.ALGO:
# A node may serve multiple purposes and so have more than 1 of these tags.
class MathsAlgorithm:
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