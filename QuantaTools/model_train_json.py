import json
import requests


def download_json(url: str) -> dict:
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def load_training_json(cfg, data : dict):
    config_data = data['Config']
    cfg.init_from_json(config_data)

    # Old json has this format    
    avg_final_loss = data['AvgFinalLoss']
    final_loss = data['FinalLoss']
    training_loss = data['TrainingLoss']
    
    if (( avg_final_loss != None ) and (avg_final_loss > 0.0)):
        cfg.avg_final_loss = avg_final_loss
    
    if (( final_loss != None ) and (final_loss > 0.0)):
        cfg.final_loss = final_loss

    return training_loss
