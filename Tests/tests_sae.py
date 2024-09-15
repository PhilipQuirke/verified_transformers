import os
import torch
import unittest
import pytest
import requests
import json
import re
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from sklearn.model_selection import ParameterGrid
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from QuantaTools.maths_tools.maths_config import MathsConfig
from QuantaTools.maths_tools.maths_data_generator import get_mixed_maths_dataloader
from QuantaTools.model_sae_train import analyze_mlp_with_sae, optimize_sae_hyperparameters


# In Visual Studo Terminal run daily:
# .\env\Scripts\activate
# As a one-off:
# pip install pytest
# pip install --upgrade jax jaxlib==0.4.1+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


class TestSae(unittest.TestCase):

    def load_model(self, num_batches):

        cfg = MathsConfig()
        cfg.model_name = "ins1_mix_d6_l3_h4_t40K_s372001" 
        cfg.parse_model_name()

        if torch.cuda.is_available():
            #print("Torch version", torch.__version__)
            #print(f"Current CUDA device: {torch.cuda.current_device()}")
            #print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

            ht_cfg = cfg.get_HookedTransformerConfig()
            cfg.main_model = HookedTransformer(ht_cfg)

            local_model_path = r"C:\Users\phili\source\repos\VerifiedArithmetic\ins1_mix_d6_l3_h4_t40K_s372001.pth"

            # Load the state dictionary
            state_dict = torch.load(local_model_path, weights_only=True)

            # Load the state dictionary into your model
            cfg.main_model.load_state_dict(state_dict)
            cfg.main_model.eval()

        dataloader = get_mixed_maths_dataloader(cfg, num_batches=num_batches, enrich_data=True)

        return cfg, dataloader


    # Test that the code runs on a small data set
    def test_sae_single(self):
        cfg, dataloader = self.load_model(num_batches=10)

        sae, score, loss, sparsity, neurons_used = analyze_mlp_with_sae(cfg, dataloader, layer_num=0, encoding_dim=64, learning_rate=0.001, sparsity_target=0.05, sparsity_weight=0.1, num_epochs=2)
        print( f"Score: {score:.4f}, Loss {loss:.4f}, Sparsity {sparsity:.4f}, Neurons Used: {neurons_used}.")


    # Can be run from Termimal with:
    # pytest -v -s -k "test_sae_sweep[full]"
    # pytest -v -s -k "test_sae_sweep[quick]"
    @pytest.mark.parametrize("full", [pytest.param(False, id="quick"), pytest.param(True, id="full")])
    def test_sae_sweep(self, full : bool = True):
        return

        if full:
            n_calls = 200
            param_space = None
        else:
            n_calls = 10
            param_space = [
                Integer(5, 6, name='encoding_dim_exp'),
                Real(1e-3, 1e-2, prior='log-uniform', name='learning_rate'),
                Real(0.01, 0.1, prior='log-uniform', name='sparsity_target'),
                Real(1e-2, 1.0, prior='log-uniform', name='sparsity_weight'),
                Real(1e-4, 1e-3, prior='log-uniform', name='l1_weight'),
                Integer(10, 12, name='num_epochs'),
                Integer(2, 3, name='patience')
            ]

        cfg, dataloader = self.load_model(num_batches=n_calls)

        save_folder = "D:\\AI\\UnitTestSae\\"
        score, file_name, params = optimize_sae_hyperparameters(cfg, dataloader, layer_num=0, param_space=param_space, save_folder=save_folder, n_calls=n_calls)

        print(f"\nBest config: File: {file_name}, Score: {score:.4f}, Parameters: {params}")
