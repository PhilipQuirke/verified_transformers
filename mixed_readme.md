What algorithm do mixed models use to perform both addition and subtraction?

Our investigation focused on ins1_mix_d6_l3_h4_t40K which is accurate. 

Some initial thoughts:
- Read section 4.3 of https://arxiv.org/pdf/2402.02619.pdf
- We assume model is reusing the addition circuits (perhaps modified) that were inserted from add_d6_l2_h3_t15K
- We assume model is accurate
- Model must pay attention to "operator" (+ or -) to understand whether to do addition or subtraction
- Model must accurately predict three distinct classes of questions:
  - Addition (answer is positive) 
  - Subtraction where D >= D' (where answer is positive)   
  - Subtraction where D < D' (where answer is negative)   
- The first answer token (A7 for 6-digit questions) is always "+" or "-" representing a positive or negative answer
  - Assuming model is accurate, A7 is accurate.
  - If A7 is "-" then operator is "-" and D < D'

Our working hypothesis is that the model's algorithm steps are:
- H1: Pays attention to the +- question operator (using OP task)
- If operator is "+" then
  - H2: Does addition using BA, MC, US & TC tasks
- Else
  - H3: Calculates whether D > D' (using NG tasks)
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

