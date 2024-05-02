# HuggingFace resources
The HuggingFace website https://huggingface.co/PhilipQuirke/VerifiedArithmetic holds the output generated for several experiments.

## Training Resources
For each model the 'VerifiedArithmeticTrain' Colab notebook generates two files:
- A "XXXXXX.pth" file containing the model weights
- A "XXXXXX_train.json" file containing configuration information and training loss data

These files are available for these models:

### 5-digit and 6-digit digit Addition models
- add_**d5_l1**_h3_t30K_s372001: Inaccurate **5-digit, 1-layer, 3-attention-head**, addition model. 
- add_d5_**l2**_h3_t15K_s372001: **Accurate** 5-digit, **2-layers**, 3-head addition model trained for 15K epochs. Training loss is 9e-9
- add_**d6**_l2_h3_t15K_s372001: **Accurate** **6-digit**, 2-layers, 3-head addition model trained for 15K epochs.  

### 6-digit Subtraction model
- **sub**_d6_l2_h3_t30K_s372001: Inaccurate 6-digit, 2-layers, 3-head **subtraction** model trained for 30K epochs.

### 6-digit Mixed (addition and subtraction) model
- **mix**_d6_l3_h4_t40K_s372001: Inaccurate 6-digit, **3-layers, 4-head mixed** (add and subtract) model trained for 40K epochs. Training loss is 8e-09

### "ins1" 6-digit Mixed models initialised with 6-digit addition model
- **ins1**_mix_d6_l3_h4_t40K_s372001: **Accurate** 6-digit, 3-layers, 4-head mixed **initialise with addition model**. Handles 1m Qs for Add and Sub. 
- **ins1**_mix_d6_l3_h4_t40K_s173289: Inaccurate. AvgFinalLoss=1.6e-08. 936K for Add, 1M Qs for Sub 
- **ins1**_mix_d6_l3_h3_t40K_s572091: Inaccurate. AvgFinalLoss=1.8e-08. Fails on 1M Qs. For 099111-099111=+0000000 gives -0000000. Improve training data.
- **ins1**_mix_d6_l3_h4_t50K_s572091: Inaccurate. AvgFinalLoss=2.9e-08. 1M for Add. 300K for Sub. For 000041-000047=-0000006 gives +0000006. Improve training data.

### "ins2" 6-digit Mixed model initialised with 6-digit addition model. Reset useful heads every 100 epochs.
- **ins2**_mix_d6_l4_h4_t40K_s372001: Inaccurate 6-digit, 3-layers, 4-head mixed initialise with addition model. **Reset useful heads every 100 epochs**. Training loss is 7e-09. Fails 1m Qs

### "ins3" 6-digit Mixed model initialised with 6-digit addition model. Reset useful heads & MLPs every 100 epochs.
- **ins3**_mix_d6_l4_h3_t40K_s372001: Inaccurate 6-digit, 3-layers, 4-head mixed initialise with addition model. **Reset useful heads and MLP every 100 epochs**. 
