# What is the mixed model algorithm?
What algorithm has a trained mixed (addition and subtraction) model learnt that allows to perform both addition and subtraction accurately?

## Initial facts and working assumptions
We investigated model ins1_mix_d6_l3_h4_t40K which answers 6-digit addition and subtraction question. Some initial thoughts:
- Read section 4.3 of https://arxiv.org/pdf/2402.02619.pdf
- For an n digit question the answer is n+2 tokens long e.g. 66666+66666=+1333332
  - We name the answer tokens An+1 to A0 
- We assume model is accurate (We know it can do 1M Qs for add and sub. This is evidence not proof of accuracy)
  - So we assume each answer token is accurate    
- The first answer token is An+1 (aka A_max, aka **sign**, A7 for 6-digit questions). It is always "+" or "-" representing a positive or negative answer
- Model must pay attention to the **operator** token (+ or -) to understand whether to do a addition or subtraction calculation
- Model must accurately predict three distinct classes of questions:
  - Addition : Answer is positive. A_max is "+". Aka ADD
  - Subtraction where D >= D' : Answer is positive. A_max is "+". Aka positive-answer-subtraction. Aka SUB 
  - Subtraction where D < D' : Answer is negative. A_max is "-". Aka negative-answer-subtraction. Aka NEG
- For subtraction questions:
  - The model must determine if D < D' before token A_max
  - D < D' is calculated as Dn < D'n or (Dn = D'n and (Dn-1 < D'n-1 or (Dn-2 = D'n-1 and ( ...
    - Addition has a parallel calculation to determine if An-1 is 1 or 0 
  - Models seem to heavily prefer "just in time" algorithms. Assume D < D' is calculated **at** the "=" token.   
- For addition questions:
  - Assume model reuses the addition circuits (perhaps modified) that were inserted from add_d6_l2_h3_t15K before the mixed model training
  - Uses tasks Base Add (BA), Make Carry 1 (MC) and Use Sum 9 (US)
  - Determines if the first numeric token of the answer (An) is 1 or 0 just in time at token position An+1

## Hypothesis 1 (Deprecated)
Our first hypothesis was that the model handles three classes of questions as follows:
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
- This mixed-model BA nodes are BA+BS nodes in the mixed model. How does that work? (Note: BS and BA give same result in edge case when D'=0 or D=D'=5. Our tests avoid this.)
- This mixed-model MC nodes are sometimes MC+BO nodes in the mixed model. How does that work?
- Are mixed-model Dn.C and Dn.Cm nodes similarily shared? TBC
- When does the model start paying attention to the operator (+/-)?
- When/how does the model calculate D > D'?
- H6 seems unlikely as it requires two passes.
  - H6 implies the model learns positive-answer-subtraction before learns negative-answer-subtraction. This seems unlikely. Models prefer to learn in parallel
  - Has model learnt a single pass approach? Seems more likely as the layer 0 attention heads are used for BA/MC and BS/BO etc. What is that method?

Overall we prefer hypothesis 2    

## Hypothesis 2
Our second hypothesis is that the model handles three classes of questions as peers:
- ADD: Addition in mixed model uses same tasks (BA, MC, US, DnCm) as addition model.
- SUB: Subtraction with positive answer uses tasks that mirror the addition tasks (BS, BO, MZ, etc)
- NEG: Subtraction with negative answer usies a third set of tasks (NS, TBA, TBA, etc)

Our second hypothesis is that the model's algorithm steps for n-digit are:
- H1: Store the question operator (+ or -)
- H2A: If operator is +, uses addition-specific TriCase, TriAdd as per Paper 2 to give Dn.C and Dn.Cm
- H2B: If operator is -, calculate if D > D' using unknown functions TBA, TBA similar toTriCase, TriAdd
- H3: Calculate A_max as : + if operator is + else + if D > D' else -
- H4: Calculate An as : Dn.Cm if operator is + else TBAn.TBAm if D > D' else TBAn.TBAm
- H5: From token position An-1, model calculates answer digit An-2 as:
  - H5A: Attention head calculates combined BA/BS/NS output
  - H5B: Attention head calculates combined MC/BO/TBA output.
  - H5C: If operator is +, (so Amax is +), attention head selects BA, MC and Dn.Cm. MLP0 layer combines to give An-2 
  - H5D: If operator is -, and Amax is +, attention head selects BS, BO and BAn.TBAm. MLP1 layer combines to give An-2
  - H5E: If operator is -, and Amax is -, attention head selects NS, UU and BAn.TBAm. MLP2 layer combines to give An-2
   
Questions/Thoughts:
- This hypothesis is more parallel (a good thing)
- This hypothesis treats ADD, SUB and NEG as three "peer" question classes (a good thing).
- We assume that the mixed model has upgraded the BA nodes to be BA/BS/NS nodes that calculate 3 "answers" for each pair of input digits. Later:
  - An addition-specific node promotes (selects) the BA answer when the operator is "+"
  - A positive-answer-subtraction-specific node promotes the BS answer when the operator is "-" and D > D'
  - A negative-answer-subtraction-specific node promotes the NS answer when the operator is "-" and D < D'
- We assume that the mixed model has upgraded the Dn.C and Dn.Cm nodes in a similar way to cope with the 3 cases
  - We assume that some nodes promotes (selects) the desired answer (paralleling the BA/BS/NS promotion technique)

## Hypothesis 2 step H5: Calculating A2
Part 27A "Calculating answer digit A2 in token position A3" in VerifiedArithmeticAnalyse.ipynb investigates Hypothesis 2 step H5

From the quanta map information:
- Two attention heads (P18L0H1 and P18L0H2) form a virtual node together and performs the A2.BA, A2.BS and A2.NS tasks.
  - Its output is used (shared) in Add, Sub and Neg question predictions.
  - TODO: How is the output data represented?   
- One attention head (P18L0H0) performs the A1.MC and A1.BO tasks.
  - TODO: Create a "NEG" version of the BO task, and test to see if this node does this task too. 
  - Its output is used (shared) in Add, Sub and (maybe) Neg question predictions.
  - TODO: How is the output data represented?
- With BA/BS/NS and MC/BO/TBA data available, the model needs perfectly accurate DnCm/TBA/TBA information:
  - Assume the perfectly accurate DnCm/TBA/TBA information is calculated in early token positions.    
  - To be accurate, at this token position (P18), the model must pull in D2C3/TBA/TBA information.
  - TODO: Which nodes pull in the D2C3/TBA/TBA information?    
- Two attention heads are specific to Add (e.g. P18L1H2, P18L1H3).
  - Both attend to the = token, which is when the sign (+ or -) is calculated.
  - TODO: Is this where the output from the P18L0H* are "filtered" to promote ADD-specific data?
  - (There are no heads, used in addition, that attend to the operator token.)
  - One head (P18L1H3) attends to A3, likely accessing information calculated in P18L0H*
- Three attention heads are specific to Sub+Neg (e.g. P18L0H3, P18L1H0, P18L1H1).
  - One head (P18L0H3) attend to the Op (+/-) token.
  - One head (P18L1H1) attends to the = token, which is when the sign (+ or -) is calculated.
  - One head (P18L1H0) attends to A3, likely accessing information calculated in P18L0H*



