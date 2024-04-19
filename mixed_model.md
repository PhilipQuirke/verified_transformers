# What is the mixed model algorithm?
We initialised a new model with an existing accurate 6-digit addition model (add_d6_l2_h3_t15K.pth) and trained it on "mixed" addition and subtraction 6-digit questions. The model (ins1_mix_d6_l3_h4_t40K.pth) can predict these questions accurately. What algorithm does it use?

## Initial facts and working assumptions
Initial thoughts on model ins1_mix_d6_l3_h4_t40K that answers 6-digit addition and subtraction question:
- Read section 4.3 of https://arxiv.org/pdf/2402.02619.pdf
- For an n digit question the answer is n+2 tokens long e.g. 66666+66666=+1333332
  - We name the answer tokens An+1 to A0 
- We assume model is 100% accurate (We know it can do 1M Qs for add and sub. This is evidence not proof.)
  - So we assume each answer token is accurate    
- Model must use the **operator** (+ or -) input token to understand whether to do a addition or subtraction calculation
- The first answer token is the **sign**.
  - This token is always "+" or "-" representing a positive or negative answer
  - This token is An+1 or A_max. For a 6-digit question it is A7. 
- Model must accurately predict three distinct classes of questions:
  - Addition : Answer is positive. A_max is "+". Aka **ADD**
  - Subtraction where D >= D' : Answer is positive. A_max is "+". Aka positive-answer-subtraction. Aka **SUB** 
  - Subtraction where D < D' : Answer is negative. A_max is "-". Aka negative-answer-subtraction. Aka **NEG**
- For subtraction questions:
  - The model must calculate if **D < D'** before token A_max
  - D < D' is calculated as Dn < D'n or (Dn = D'n and (Dn-1 < D'n-1 or (Dn-2 = D'n-1 and ( ...
    - Addition has a similar calculation (TriCase, TriAdd) to calculate if An-1 is 1 or 0 
  - Our models seem to heavily prefer "just in time" algorithms. Assume D < D' is calculated **at** the "=" token.   
- For addition questions, assume the model:
  - Reuses the addition circuits (perhaps modified) that were inserted from add_d6_l2_h3_t15K before
  - Re-uses tasks Base Add (BA), Make Carry 1 (MC), Use Sum 9 (US), TriCase (C) and TriAdd (Cn) sub-tasks
  - Determines if the first numeric token of the answer (An) is 1 or 0 just in time at token position An+1

## Hypothesis 1 (Deprecated)
Our first hypothesis was that the mixed model handles three classes of questions as follows:
- ADD: Addition in this mixed model uses the same tasks (BA, MC, US, DnCm) as addition model.
- SUB: Subtraction with a positive answer uses tasks that mirror the addition tasks (BS, BO, MZ, etc)
- NEG: Subtraction with a negative answer uses the mathematics rule A-B = -(B-A), uses above case to do the bulk of the work 

Specifically, our hypothesis is that the model's algorithm steps are:
- H1: Pays attention to the +- question operator (using OP task)
- If operator is "+" then
  - H2: Does addition using BA, MC, US & TC tasks
- Else
  - H3: Calculates if D > D' (using NG nodes)
  - If D > D' then
    - H4: Amax is "+"
    - H4: Does subtraction using BS, BO, SZ & T?? tasks
  - Else
    - H5: Amax is =-"
    - H6A: Applys D - D' = - (D' - D) transform
    - H4: Does subtraction using BS, BO, SZ & T?? tasks
    - H6B: Applys D - D' = - (D' - D) transform a second time

Questions/Thoughts:
- H6 seems unlikely as it requires two passes.
  - H6 implies the model learns positive-answer-subtraction before learns negative-answer-subtraction. This seems unlikely. Models prefer to learn in parallel
  - Has model learnt a single pass approach? Seems more likely as the layer 0 attention heads are used for BA/MC and BS/BO etc. What is that method?

Overall we prefer hypothesis 2    

## Hypothesis 2
Our current hypothesis is that the model handles three classes of questions as peers:
- **ADD:** Addition in mixed model uses same tasks (BA, MC, US, DnCm) as addition model.
- **SUB:** Subtraction with positive answer uses tasks that mirror the addition tasks 
- **NEG:** Subtraction with negative answer uses a third set of tasks 

This lead us to change the Paper 1 sub-task abbreviations to give a coherent naming convention across the 3 question classes:
![Hypo2_A2_Terms](./assets/Hypothesis2_Terminology.png?raw=true "Hypothesis 2 Terminology")

TODO: In Paper 2, for consistency, update the text and all diagrams containing this terminology.

Our current hypothesis is that the model's algorithm steps for n-digit are:
- H1: Store the question operator **OPR** (+ or -)
- H2A: If OPR is +, uses addition-specific TriCase, TriAdd as per Paper 2 to give Dn.ST and Dn.STm (previously called Dn.C and Dn.CM)
- H2B: If OPR is -, calculate if D > D' using functions MT (similar to addition's ST function)
- H3: Calculate the sign **SGN** (aka A_max) as : + if OPR is + else + if D > D' else -
- H4: Calculate An as : Dn.STm if OPR is + else Dn.MTm if D > D' else Dn.NTm
- H5: From token position An-1, model calculates answer digit An-2 as:
  - H5A: Attention head calculates combined SA/MD/ND output
  - H5B: Attention head calculates combined SC/MB/NB output.
  - H5C: If OPR is +, (so SGN is +), attention head selects SA, SC and Dn.STm. MLP0 layer combines to give An-2 
  - H5D: If OPR is -, and SGN is +, attention head selects MD, MB and Dn.MTm. MLP1 layer combines to give An-2
  - H5E: If OPR is -, and SGN is -, attention head selects ND, NB and Dn.NTm. MLP2 layer combines to give An-2
   
Questions/Thoughts:
- This hypothesis is more parallel (a good thing)
- This hypothesis treats ADD, SUB and NEG as three "peer" question classes (a good thing).
- Assume the mixed model learnt to "upgrade" the initialised SA nodes to be SA/MD/ND nodes that calculate 3 "answers" for each pair of input digits:
  - Another addition-specific node promotes (selects) the SA answer when the OPR is "+"
  - A positive-answer-subtraction-specific node promotes the MD answer when the OPR is "-" and D > D'
  - A negative-answer-subtraction-specific node promotes the ND answer when the OPR is "-" and D < D'
  - TODO: How is the output data represented?
- Assume the mixed model learnt to "upgrade" the initialised Dn.STm nodes to be STm/MTm/NTm nodes that calculate 3 "answers" for each pair of input digits:
  - Another node promotes (selects) the desired answer 
  - TODO: How is the output data represented?

## Hypothesis 2 step H5: Calculating A2
Part 27A "Calculating answer digit A2 in token position A3" in VerifiedArithmeticAnalyse.ipynb investigates Hypothesis 2 step H5 generating this quanta map:

![A2QuantaMap](./assets/ins1_mix_d6_l3_h4_t40K_s372001QuantaAtP18.svg?raw=true "A2 Quanta Map")

From this quanta map, we see:
- Two attention heads (P18L0H1 and P18L0H2) form a virtual node together and performs the A2.SA, A2.MD and A2.ND tasks.
  - Its output is used (shared) in Add, Sub and Neg question predictions.
  - TODO: How is the output data represented?
- The SS (Use Sum 9), MZ and NZ (Sum to Zero) sub-tasks are **not** used in this model:
  - The model has optimised them out - It now relies on the accurate Dn.STm/MTm/NTm values instead  
  - (A Paper 2 deprecated hypothesis for the addition model describes this possibility)
- The SC (Carry 1), MB and NB (Borrow one) sub-tasks are used in **some** answer digits: 
  - The model has optimised out some instances - It now relies on the accurate Dn.STm/MTm/NTm values instead  
  - (A Paper 2 deprecated hypothesis for the addition model describes this possibility)
- One attention head (P18L0H0) performs the A1.SC, A1.MB and A1.NB tasks.
  - Its output is used (shared) in Add, Sub and Neg question predictions.
  - TODO: How is the output data represented?- 
- The model needs perfectly accurate Dn.STm/MTm/NTm information:
  - Assume the perfectly accurate Dn.STm/MTm/NTm information is calculated in early token positions.    
  - To be accurate, at the P18 token position, the model must pull in D2.ST3/MT3/NT3 information.
  - TODO: Which nodes pull in the D2.ST3/MT3/NT3 information?    
- Two attention heads are specific to Add (e.g. P18L1H2, P18L1H3).
  - Both attend to the = token, which is when the sign (+ or -) is calculated.
  - TODO: Is this where the output from the P18L0H* are "filtered" to promote ADD-specific data?
  - (There are no heads, used in addition, that attend to the OPR token.)
  - One head (P18L1H3) attends to A3, likely accessing information calculated in P18L0H*
- Three attention heads are specific to Sub+Neg (e.g. P18L0H3, P18L1H0, P18L1H1).
  - One head (P18L0H3) attend to the OPR (+/-) token.
  - One head (P18L1H1) attends to the = token, which is when the sign (+ or -) is calculated.
  - One head (P18L1H0) attends to A3, likely accessing information calculated in P18L0H*

Our hypothesis is that to predict A2 the model's algorithm steps, broken down by question class, are:
![Hypo2_A2Calc](./assets/Hypothesis2_A2_Calc.png?raw=true "Hypothesis2 A2 Calc")

