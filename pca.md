# Principal Component Analysis 

## Definition
Principal Component Analysis (PCA) is a powerful statistical technique used for dimensionality reduction in machine learning. 
It aids in mechanistic interpretability by simplifying complex datasets into principal components that capture the most significant variance within the data. 
By transforming the original data into a smaller set of uncorrelated variables, PCA allows for the easier visualization and understanding of the underlying mechanisms influencing the dataset. 
This reduction highlights the most influential features and also helps in uncovering the intrinsic structure of the data, facilitating a more intuitive interpretation of the mechanisms driving the observed phenomena. 

## Quanta Tools
This library uses PCA to help understand the purpose of individual useful nodes. 
If a researcher suspects a useful node has a specific output pattern corresponding to specific classes of input, they can test this by creating input questions, asking the model to predict the answers, and then do a PCA analysis of the output. 
If the PCA says the node's output is strongly clustered, aligned to the input question classes, then we conclude the node is interpretable, and we add a "PCA" tag to the node as documentation (e.g. "PCA.A0.PA" )     

## Example usage
The Increasing Trust in Language Models through the Reuse of Verified Circuits paper ( https://arxiv.org/abs/2402.02619 ) says that some nodes in the Addition model perform "TriCase" calculations - converting 3 classes of input questions into 2 or 3 output bits. 

In VerifiedArithmeticTrain.ipynb Colab notebook, we pick a single useful node (e.g. P19L2H1) and test it:
- We construct 100 test questions for each of the 3 input classes (via qt.make_maths_tricase_questions)
- We ask the model to predict the answers to the 300 questions
- We run PCA on the output of the model (via calc_pca_for_an)
- If the PCA says node's output is clustered into 2 or 3 clusters, aligned to the input question classes, then we conclude the node is interpretable, and we add a "PCA" tag to the node as documentation (e.g. "PCA.A0.PA" )      

The library runs the PCA over many nodes, adding tags to interpretable nodes, and producing output like this:  

![PcaResults](./assets/ins1_mix_d6_l3_h4_t40K_s372001PcaTr.svg?raw=true "PCA Results")
