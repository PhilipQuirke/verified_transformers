# verified_transformers
Tool used to verify accuracy of transformer model. Contains:

- Notebooks: Colab notebook files
  - VerifiedArithmeticTrain: Colab used to train transformer arithmetic models. 
    - Outputs pth and json files that are (manually) stored on HuggingFace
  - VerifiedArithmeticAnalyse: Colab used to analyze the behavior and algorithm of transformer (pre-trained) arithmetic models
      - Inputs pth files from training phase from HuggingFace
      - Outputs *_behavior and *_algorithm json files that are (manually) stored on HuggingFace
      
- QuantaTools: Python library code
  - model_*.py: Contains the configuration of the transformer model being trained/analysed
  - useful_*.py: Contains data on the useful token positions and useful nodes (attention heads and MLP neurons) that the model uses in predictions
  - quanta_*.py: Contains categorisations of model behavior (aka quanta), with ways to detect and grrpah them. Applicable to all models
  - maths_*.py: Cotains specializations of the above specific to arithmetic (addition and subctraction) transformer models
          
- Tests: Unit tests 
