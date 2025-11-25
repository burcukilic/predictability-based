import torch
from models import EffectPredictorDeepSym, SIU_Deepsym
import argparse
import yaml
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate the models")
    parser.add_argument("-m", "--model_folder", type=str, help="Model Folder")
    parser.add_argument("-d", "--data_folder", type=str,  help="Data Folder")
    parser.add_argument("-f", "--format", type=str, default="last", help="Model format")
    args = parser.parse_args()

    
    device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model config
    config_file = "./logs/" + args.model_folder + "/config.yaml"

    config = yaml.safe_load(open(config_file, "r"))
    del config['lr']
    if args.model_folder == "random1" or args.model_folder == "active1":
        model = SIU_Deepsym(**config)
    else:    
        model = EffectPredictorDeepSym(**config)
    model = model.to(device)
    model.load_state_dict(torch.load("logs/"+args.model_folder+"/" + args.format + ".ckpt", map_location=device))
    
    actions = torch.load("data/"+args.data_folder+"/action.pt", map_location=device).to(device)
    states = torch.load("data/"+args.data_folder+"/state.pt"  ,map_location=device).to(device)
    effects = torch.load("data/"+args.data_folder+"/effect.pt",map_location=device).to(device)

    model.eval()
    if args.format != "last":
        actions = actions[:int(args.format)]
        states = states[:int(args.format)]
        effects = effects[:int(args.format)]

    with torch.no_grad():
        if args.model_folder == "random1" or args.model_folder == "active1":
            predictions, _, _ = model(states, actions)
            
        else:
            predicted_distribution, _, _ = model(states, actions)
            predictions = predicted_distribution.loc

            print("Predictions: ", predictions.abs().min(dim=0).values, predictions.abs().max(dim=0).values)
            print("Effects:     ", effects.abs().min(dim=0).values, effects.abs().max(dim=0).values)

            print("Nulls in predictions: ", predictions[torch.norm(predictions.abs(), dim=1) < 0.05].shape[0], predictions[torch.norm(predictions.abs(), dim=1) < 0.05].shape[0]/predictions.shape[0])
            print("Nulls in effects:     ", effects[torch.norm(effects.abs(), dim=1) < 0.05].shape[0], effects[torch.norm(effects.abs(), dim=1) < 0.05].shape[0]/effects.shape[0])
        
        abs_error = torch.abs(predictions - effects)
        print("Max Error: ", abs_error.max(dim=0).values)
        print("Min Error: ", abs_error.min(dim=0).values)
        print("Mean Error: ", abs_error.mean(dim=0))
        print("Std Error: ", abs_error.std(dim=0))
