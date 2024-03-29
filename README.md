# verified_transformers
This library contains tool to help train various transformer models trained to perform integer addition, subtraction and multiplication e.g. 133357+182243=+0315600, 123450-345670=-0123230 and 000345*000823=+283935. Each digit is a separate token. For 6 digit questions, the model is given 14 "question" (input) tokens, and must then predict the corresponding 8 "answer" (output) tokens.

The library contains tools to help investigate, graph and understand the transformer model's agorithm. 
Sample automatically-generated tool images are available in https://github.com/PhilipQuirke/verified_transformers/tree/main/assets

This library contains files:

- **Notebooks:** Jupyter notebooks which are run in Goole Colab: 
  - VerifiedArithmeticTrain.ipynb: Colab used to train transformer arithmetic models. 
    - Outputs pth and json files that are (manually) stored on HuggingFace
  - VerifiedArithmeticAnalyse.ipynb: Colab used to analyze the behavior and algorithm of transformer arithmetic models
    - Inputs pth files (generated above) from HuggingFace
    - Outputs *_behavior and *_algorithm json files that are (manually) stored on HuggingFace 
  - Accurate_Math_Train.ipynb: Deprecated. Predecessor of VerifiedArithmeticTrain associated with https://arxiv.org/abs/2402.02619 
  - Accurate_Math_Analyse.ipynb: Deprecated. Predecessor of VerifiedArithmeticAnalyse associated with https://arxiv.org/abs/2402.02619

- **QuantaTools:** Python library code imported into the notebooks:
  - model_*.py: Contains the configuration of the transformer model being trained/analysed
  - useful_*.py: Contains data on the useful token positions and useful nodes (attention heads and MLP neurons) that the model uses in predictions
  - quanta_*.py: Contains categorisations of model behavior (aka quanta), with ways to detect and graph them 
  - ablate_*.py: Contains ways to "intervention ablate" the model and detect the impact of the ablation
  - algo_*.py: Contains tools to support declaring and validating a model algorithm
  - maths_*.py: Contains specializations of the above specific to arithmetic (addition and subtraction) transformer models
          
- Tests: Unit tests 
          
**HuggingFace** permanently stores the output files generated by the 'train' and 'analyse' Colabs:
- VerifiedArithmeticTrain/Analyse files are stored at https://huggingface.co/PhilipQuirke/VerifiedArithmetic covering these models:
  - add_**d5_l1**_h3_t30K: Inaccurate **5-digit, 1-layer, 3-attention-head**, addition model. 
  - add_d5_**l2**_h3_t15K: **Accurate** 5-digit, **2-layers**, 3-head addition model trained for 15K epochs. Training loss is 9e-9
  - add_**d6**_l2_h3_t15K: **Accurate** **6-digit**, 2-layers, 3-head addition model trained for 15K epochs.  
  - **sub**_d6_l2_h3_t30K: Inaccurate 6-digit, 2-layers, 3-head **subtraction** model trained for 30K epochs.
  - **mix**_d6_l3_h4_t40K: Inaccurate 6-digit, **3-layers, 4-head mixed** (add and subtract) model trained for 40K epochs. Training loss is 8e-09
  - **ins1**_mix_d6_l3_h4_t40K: **Accurate** 6-digit, 3-layers, 4-head mixed **initialise with addition model**. Handles 1m Qs for Add and Sub. 
  - **ins2**_mix_d6_l4_h4_t40K: Inaccurate 6-digit, 3-layers, 4-head mixed initialise with addition model. **Reset useful heads every 100 epochs**. Training loss is 7e-09. Fails 1m Qs
  - **ins3**_mix_d6_l4_h3_t40K: Inaccurate 6-digit, 3-layers, 4-head mixed initialise with addition model. **Reset useful heads and MLP every 100 epochs**. 
- Accurate_Math_Train/Analyse (deprecated) files are stored at https://huggingface.co/PhilipQuirke/Accurate5DigitAddition

The associated **papers** are:
- Understanding Addition in Transformers: https://arxiv.org/abs/2310.13121 . Model add_d5_l1_h3_t30K is very similar to the one in this paper.
- Increasing Trust in Language Models through the Reuse of Verified Circuits. https://arxiv.org/abs/2402.02619. Uses many of these models mostly focusing on add_d5_l2_h3_t15K, add_d6_l2_h3_t15K and ins1_mix_d6_l3_h4_t40K
- The next paper will focus on explaining model ins1_mix_d6_l3_h4_t40K
