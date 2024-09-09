import os
import torch
import unittest
import pytest
import requests
import json
import re
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

from QuantaTools.maths_tools.maths_config import MathsConfig
from QuantaTools.maths_tools.maths_data_generator import get_mixed_maths_dataloader
from QuantaTools.model_sae_train import analyze_mlp_with_sae, optimize_sae_hyperparameters


class TestSae(unittest.TestCase):

    def load_model(self):

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

        dataloader = get_mixed_maths_dataloader(cfg, num_batches=1000, enrich_data=True)
        #print("Data set size", len(dataloader.dataset))  

        return cfg, dataloader


    def test_sae_single(self):
        cfg, dataloader = self.load_model()

        sae, score, loss, sparsity, neurons_used = analyze_mlp_with_sae(cfg, dataloader, layer_num=0, encoding_dim=64, learning_rate=0.001, sparsity_target=0.05, sparsity_weight=0.1, num_epochs=10)
        print( f"Score: {score:.4f}, Loss {loss:.4f}, Sparsity {sparsity:.4f}, Neurons Used: {neurons_used}.")


    # Can be run from Termimal with:
    # pytest -v -s -k "test_sae_sweep[full]"
    # pytest -v -s -k "test_sae_sweep[quick]"
    @pytest.mark.parametrize("full", [pytest.param(False, id="quick"), pytest.param(True, id="full")])
    def test_sae_sweep(self, full : bool = False):

        if full:
            cfg, dataloader = self.load_model()

            param_grid = {
                'encoding_dim': [32, 64, 128, 256, 512],
                'learning_rate': [1e-4, 1e-3, 1e-2],
                'sparsity_target': [0.01, 0.05, 0.1, 0.25],
                'sparsity_weight': [1e-2, 1e-1, 1.0],
                'l1_weight': [1e-4, 1e-3, 1e-2], 
                'num_epochs': [10],
                'patience': [2]
            }

            num_experiments = 1
            for param_values in param_grid.values():
                num_experiments *= len(param_values)

            print(f"Number of configurations to test: {num_experiments}")

            # This takes > 1 hour to run
            save_folder = "D:\\AI\\TRAINSAE\\"
            sae, score, neurons_used, params = optimize_sae_hyperparameters(cfg, dataloader, layer_num=0, param_grid=param_grid, save_folder=save_folder)