import torch
import unittest
import requests
import json
from bs4 import BeautifulSoup
import re
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download

from QuantaTools.model_token_to_char import token_to_char, tokens_to_string
from QuantaTools.model_train_json import download_huggingface_json, load_training_json

from QuantaTools.useful_node import NodeLocation, UsefulNode, UsefulNodeList

from QuantaTools.quanta_map_impact import sort_unique_digits, get_quanta_impact
from QuantaTools.quanta_map_attention import get_quanta_attention

from QuantaTools.maths_tools.maths_config import MathsConfig
from QuantaTools.maths_tools.maths_constants import MathsToken, MathsBehavior
from QuantaTools.maths_tools.maths_utilities import set_maths_vocabulary, int_to_answer_str, tokens_to_unsigned_int



class TestHuggingFace(unittest.TestCase):


    def get_training_json_files(self, repo_id):
        api = HfApi()
        files = api.list_repo_files(repo_id)
        return [f for f in files if f.endswith('_train.json')]


    def test_hugging_face_training_data(self):
        repo_id = "PhilipQuirke/VerifiedArithmetic"

        # Get list of training json files
        train_files = self.get_training_json_files(repo_id)
        self.assertGreater(len(train_files), 5)

        # Process each file
        for filename in train_files:
            data = download_huggingface_json(repo_id, filename)

            # Can we load the training json file?
            cfg = MathsConfig()
            load_training_json(cfg, data)
            self.assertGreater(cfg.avg_final_loss, 0)
            self.assertGreater(cfg.final_loss, 0)  
            