What algorithm has a trained mixed model learnt that allows to perform both addition and subtraction accurately?

We investigation model ins1_mix_d6_l3_h4_t40K which answers 6-digit question. Some initial thoughts:
- Read section 4.3 of https://arxiv.org/pdf/2402.02619.pdf
- For an n digit question the answer is n+2 tokens long e.g. 66666+66666=+1333332
  - We name the answer tokens An+1 to A0 
- We assume model is accurate (We know it can do 1M Qs for add and sub but this evidence not proof accuracy)
  - So we are assuming each answer token is accurate    
- The first answer token An+1 (A7 for 6-digit questions) is always "+" or "-" representing a positive or negative answer
- Model must pay attention to the **operator** token (+ or -) to understand whether to do a addition or subtraction calculation
- Model must accurately predict three distinct classes of questions:
  - Addition (Answer is positive. An+1 is "+") 
  - Subtraction where D >= D' (Answer is positive. An+1 is "+")    
  - Subtraction where D < D' (Answer is negative. An+1 is "-")
- For subtraction questions:
  - The model must determine if D < D' before token A_max
  - D < D' is calculated as Dn < D'n or (Dn = D'n and (Dn-1 < D'n-1 or (Dn-2 = D'n-1 and ( ...
    - (Addition has a parallel calculation to determine if An-1 is 1 or 0 
  - Models seem to always have "just in time" algorithms. So assume D < D' is calculated **at** the "=" token.   
- For addition questions:
  - We assume model reuses the addition circuits (perhaps modified) that were inserted from add_d6_l2_h3_t15K before the mixed model training
  - Uses tasks Base Add (BA), Make Carry 1 (MC) and Use Sum 9 (US)
  - Determines if the first numeric token of the answer (An) is 1 or 0 just in time at token position An+1

Our first hypothesis was that the model's algorithm steps are:
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
Questions:
- The insert-model BA nodes are BA+BS nodes in the mixed model. How does that work? (Note: BS and BA give same result in edge case when D'=0 or D=D'=5. Our tests avoid this.)
- The insert-model MC nodes are sometimes MC+BO nodes in the mixed model. How does that work?
- Are insert-model TC nodes similarily shared? TBC
- When does the model start paying attention to the operator (+/-)?
- When/how does the model calculate D > D'?
- H6 seems unlikely as it requires two passes. Has model learnt a single pass approach? Seems more likely as the layer 0 attention heads are used for BA/MC and BS/BO etc. What is that method?

Our second hypothesis is that the model's algorithm steps are:
- H1: Store the question operator
- H2: If operator is -, calculates if D > D' using unknown functions XXX, YYY, ZZZ
- H3: If operator is +, calculates TriCase, TriAdd etc to give Dn.Cm
- H4: Calculate An+1 as : + if operator is + else + if D > D' else -
- H5: Calculate An as : Dn.Cm if operator is + else XXXn.YYYm if D > D' else XXXn.ZZZm
- H6: From An-1, model calculates An-2 as:
  - H7: Attention head calculates combined BA/BS/TT output
  - H8: Attention head calculates combined MC/BO/UU output.
  - H9: If operator is +, attention head selects BA, MC and Dn.Cm. MLP0 layer combines to give An-2 
  - H10: If operator is -, and D > D', attention head selects BS, BO and XXXn.YYYm. MLP1 layer combines to give An-2
  - H11: If operator is -, and not D > D', attention head selects TT, UU and XXXn.ZZZm. MLP2 layer combines to give An-2
   
