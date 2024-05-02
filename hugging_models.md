# HuggingFace resources
The HuggingFace website https://huggingface.co/PhilipQuirke/VerifiedArithmetic holds the output generated for several experiments.

## Training Resources
For each model the 'VerifiedArithmeticTrain' Colab notebook generates two files:
- A "XXXXXX.pth" file containing the model weights
- A "XXXXXX_train.json" file containing configuration information and training loss data

These files are available for these models:

### 5-digit and 6-digit digit Addition models
- add_d5_l1_h3_t30K_s372001: Inaccurate **5-digit, 1-layer, 3-attention-head**, addition model. Reproduces Paper 1 model. Can predict S0, S1 and S2 complexity addition questions.
- add_d5_l2_h3_t15K_s372001: **Accurate** 5-digit, **2-layers**, 3-head addition model trained for 15K epochs. Training loss is 9e-9
- add_d6_l2_h3_t15K_s372001: **Accurate** **6-digit**, 2-layers, 3-head addition model trained for 15K epochs.  

### 6-digit Subtraction model
- sub_d6_l2_h3_t30K_s372001: Inaccurate 6-digit, 2-layers, 3-head subtraction model trained for 30K epochs.

### 6-digit Mixed (addition and subtraction) model
- mix_d6_l3_h4_t40K_s372001: Inaccurate 6-digit, **3-layers, 4-head mixed** (add and subtract) model trained for 40K epochs. Training loss is 8e-09

### "ins1" 6-digit Mixed models initialised with 6-digit addition model
- ins1_mix_d6_l3_h4_t40K_s372001: **Accurate** 6-digit, 3-layers, 4-head mixed initialised with addition model. Handles 1m Qs for Add and Sub. 
- ins1_mix_d6_l3_h4_t40K_s173289: Inaccurate. AvgFinalLoss=1.6e-08. 936K for Add, 1M Qs for Sub 
- ins1_mix_d6_l3_h3_t40K_s572091: Inaccurate. AvgFinalLoss=1.8e-08. Fails on 1M Qs. For 099111-099111=+0000000 gives -0000000. Improve training data.
- ins1_mix_d6_l3_h4_t50K_s572091: Inaccurate. AvgFinalLoss=2.9e-08. 1M for Add. 300K for Sub. For 000041-000047=-0000006 gives +0000006. Improve training data.

### "ins2" 6-digit Mixed model initialised with 6-digit addition model. Reset useful heads every 100 epochs.
- ins2_mix_d6_l4_h4_t40K_s372001: Inaccurate 6-digit, 3-layers, 4-head mixed initialise with addition model. **Reset useful heads every 100 epochs**. Training loss is 7e-09. Fails 1m Qs

### "ins3" 6-digit Mixed model initialised with 6-digit addition model. Reset useful heads & MLPs every 100 epochs.
- ins3_mix_d6_l4_h3_t40K_s372001: Inaccurate 6-digit, 3-layers, 4-head mixed initialise with addition model. **Reset useful heads and MLP every 100 epochs**. 

## Analysis Resources
For each model the 'VerifiedArithmeticAnalysis' Colab notebook generates two files:
- A "XXXXXX_behavior.json" file containing "behavior" facts automatically learnt about the model 
- A "XXXXXX_maths.json" file containing "maths-specific" facts automatically learnt about the model  

### Behavior json example
The ins1_mix_d6_l3_h4_t40K_s372001_behavior.json file starts with:
```
[{"position": 0, "layer": 0, "is_head": true, "num": 3, "tags": ["Fail%:5", "Impact:A7", "Math.Sub:M0", "Math.Neg:N1234", "Attn:P0=100"]}, 
{"position": 6, "layer": 0, "is_head": true, "num": 0, "tags": ["Fail%:2", "Impact:A43210", "Math.Sub:M123", "Math.Neg:N1", "Attn:P1=73", "Attn:P6=23", "Attn:P4=2", "Attn:P2=1"]}, 
{"position": 9, "layer": 0, "is_head": true, "num": 0, "tags": ["Fail%:3", "Impact:A765", "Math.Add:S1234", "Math.Sub:M3", "Math.Neg:N1", "Attn:P8=50", "Attn:P1=49"]}, 
{"position": 9, "layer": 0, "is_head": true, "num": 1, "tags": ["Fail%:1", "Impact:A654", "Math.Add:S124", "Attn:P8=52", "Attn:P1=46"]}, {"position": 9, "layer": 0, "is_head": false, "num": 0, "tags": ["Fail%:8", "Impact:A765", "Math.Add:S12345",
```

### Maths json example
Some lines of the ins1_mix_d6_l3_h4_t40K_s372001_maths.json file are:
```
[{"position": 0, "layer": 0, "is_head": true, "num": 3, "tags": []},
{"position": 6, "layer": 0, "is_head": true, "num": 0, "tags": ["Algo:OPR"]},
{"position": 9, "layer": 0, "is_head": true, "num": 0, "tags": ["Algo:D4.GT"]},
...
{"position": 15, "layer": 0, "is_head": true, "num": 2, "tags": ["Algo:A5.SA", "Algo:A5.MD", "Algo:A5.ND.A5"]},
{"position": 15, "layer": 0, "is_head": true, "num": 3, "tags": ["Algo:OPR", "Algo:SGN"]},
{"position": 15, "layer": 0, "is_head": false, "num": 0, "tags": []}, {"position": 15, "layer": 1, "is_head": true, "num": 1, "tags": []}, {"position": 15, "layer": 1, "is_head": true, "num": 2, "tags": []}, {"position": 15, "layer": 1, "is_head": true, "num": ...
```
