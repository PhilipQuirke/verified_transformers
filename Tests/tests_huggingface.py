import torch
import unittest
import requests
import json
from bs4 import BeautifulSoup
import re

from QuantaTools.model_token_to_char import token_to_char, tokens_to_string
from QuantaTools.model_train_json import download_json, load_training_json

from QuantaTools.useful_node import NodeLocation, UsefulNode, UsefulNodeList

from QuantaTools.quanta_map_impact import sort_unique_digits, get_quanta_impact
from QuantaTools.quanta_map_attention import get_quanta_attention

from QuantaTools.maths_tools.maths_config import MathsConfig
from QuantaTools.maths_tools.maths_constants import MathsToken, MathsBehavior
from QuantaTools.maths_tools.maths_utilities import set_maths_vocabulary, int_to_answer_str, tokens_to_unsigned_int



class TestHuggingFace(unittest.TestCase):


    def get_training_json_files(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        files = []
        for link in soup.find_all('a', href=re.compile(r'.*_train\.json$')):
            files.append(link['href'])
        return files


    def test_hugging_face_training_data(self):
        # URL of the repository
        repo_url = "https://huggingface.co/PhilipQuirke/VerifiedArithmetic/raw/main/"

        # Get list of train files
        train_files = self.get_training_json_files(repo_url)

        # Process each file
        for file in train_files:
            file_url = f"https:/{file}"
            data = download_json(file_url)
    
            # Can we load the training json file?
            cfg = MathsConfig()
            load_training_json(cfg, data)
            self.assertGreater( cfg.avg_final_loss, 0)
            self.assertGreater( cfg.final_loss, 10000) # Should fail
            