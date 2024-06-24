import json
import requests
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download


def download_huggingface_json(repo_id, filename):
    file_path = hf_hub_download(repo_id=repo_id, filename=filename) 
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data


def load_training_json(cfg, data : dict):
    config_data = data['Config']
    cfg.init_from_json(config_data)

    # Old json has this format    
    avg_final_loss = data.get('AvgFinalLoss', 0)
    final_loss = data.get('FinalLoss', 0)
    training_loss = data['TrainingLoss']
    
    if (( avg_final_loss != None ) and (avg_final_loss > 0.0)):
        cfg.avg_final_loss = avg_final_loss
    
    if (( final_loss != None ) and (final_loss > 0.0)):
        cfg.final_loss = final_loss

    return training_loss
