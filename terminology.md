# Terminology

These terms and abbreviations are used in Paper 2, this repository's Colabs and python code:

- Pn : Model (input or output) token position. Zero-based. e.g. **P**18, **P**18L1H0
- Ln : Model layer n. Zero-based. e.g. add_d6_**l**2_h3_t15K, P18**L**1H2
- Hn : Attention head n. Zero-based. e.g. add_d6_l2_**h**3_t15K, P18L1**H**2
- Mn : MLP neuron n. Zero-based
- PnLnHn : Location / name of a single attention head, at a specified layer, at a specific token position
- PnLnMn : Location / name of a single MLP neuron, at a specified layer, at a specific token position
- D : First number of the pair question numbers
- Dn : nth numeric token in the first question number. Zero-based. D0 is the units value
- D' : Second number of the pair question numbers
- D'n : nth token in the second question number. Zero-based. D0 is the units value
- A : Answer to the question (including answer sign)
- An : nth token in the answer. Zero-based. A0 is the units value. The highest token is the "+" or "-" answer sign
- Amax : The highest token in the answer. It is always the "+" or "-" sign
- S : Prefix for Addition. Think S for Sum. Aka ADD.
- SA : Basic Add. An addition sub-task. An.SA is defined as (Dn + D'n) % 10. For example, 5 + 7 gives 2
- SC : Make Carry. An addition sub-task. An.SC is defined as Dn + D'n >= 10. For example, 5 + 7 gives True
- SS : Make Sum 9. An addition sub-task. An.SS is defined as Dn + D'n == 9. For example, 5 + 7 gives False
- ST : TriCase. An addition sub-task. Refer paper 2 for details
- ST8, ST9, ST10: Outputs of the ST TriCase sub-task. Refer paper 2 for details
- M : Prefix for Subtraction with a positive answer. Think M for Minus. Aka SUB
- MD: Basic Difference. A subtraction sub-task. An.MD is defined as (Dn - D'n) % 10. For example, 3 - 7 gives 6
- MB: Borrow One. A positive-answer subtraction sub-task. An.MB is defined as Dn - D'n < 0. For example, 5 - 7 gives True
- MZ : Make Zero. A positive-answer subtraction sub-task. An.MZ is defined as Dn - D'n == 0. For example, 5 - 5 gives True
- MT : TriCase. A positive-answer subtraction sub-task. Refer paper 2 for details
- MT1, MT0, MT-1: Outputs of the MT TriCase sub-task. Refer paper 2 for details
- N : Prefix for Subtraction with a negative answer. Think N for Negative. Aka NEG
- ND: Basic Difference. A negative-answer subtraction sub-task. An.ND is defined as (Dn - D'n) % 10. For example, 3 - 7 gives 6
- NB: Borrow One. A negative-answer subtraction sub-task. An.NB is defined as Dn - D'n < 0. For example, 5 - 7 gives True
- NZ : Make Zero. A negative-answer subtraction sub-task. An.NZ is defined as Dn - D'n == 0. For example, 5 - 5 gives True
- NT : TriCase. A negative-answer subtraction sub-task. Refer paper 2 for details
- GT : Greater Than. A (positive-answer or negative-answer) subtraction sub-task. Dn.GT is defined as Dn > D'n. For example, 3 > 5 gives False
- OPR: Operator. A sub-task that attends to the + or - token in the question (which determines whether the question is addition or subtraction).
- SGN: Sign. A sub-task that attends to the first answer token, which is + or -
- PCA: Principal Component Analysis
- EVR: Explained Variance Ratio. In PCA, EVR represents the percentage of variance explained by each of the selected components.
