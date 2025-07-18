import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm
from models import EffectPredictorDeepSym
from torch.utils.data import DataLoader, TensorDataset
import argparse
import yaml

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train the model")
    parser.add_argument("-o", "--output", type=str, default="dump", help="Output directory")
    parser.add_argument("-e", "--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("-m", "--model", type=str, help="Model folder name")
    parser.add_argument("-jid", "--data-jid", type=str, help="Data Folder")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    data_folder = "./data/" + args.data_jid

    actions = (torch.load(data_folder + "/action.pt").to(device))
    states  = (torch.load(data_folder + "/state.pt").to(device))
    effects = (torch.load(data_folder + "/effect.pt").to(device))
    
    dataset = TensorDataset(states, actions, effects)

    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    # Load the model config
    config_file = "./logs/" + args.model + "/config.yaml"

    config = yaml.safe_load(open(config_file, "r"))

    state_dim = config["state_dim"]
    action_dim = config["action_dim"]
    hidden_dim = config["hidden_dim"]
    output_dim = config["output_dim"]
    state_latent_dim = config["state_latent_dim"]
    action_latent_dim = config["action_latent_dim"]
    activation = "sigmoid"#config["activation"]

    model = EffectPredictorDeepSym(state_dim, action_dim, hidden_dim, output_dim, state_latent_dim, action_latent_dim, deterministic=False, activation=activation)
    model = model.to("cuda")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    losses = []
    nll_losses = []
    nce_losses = []
    entropies = []
    nce_states = []
    nce_actions = []

    for epoch in tqdm(range(args.epochs)):
        
        total_loss       = 0
        total_nll_loss   = 0
        total_nce_loss   = 0
        total_spar_loss  = 0
        total_state_nce  = 0
        total_action_nce = 0

        if epoch > 0 and epoch%30 == 0:
            model.validate_emb()
            model.validate_actions()

        for batch in dataloader:
            optimizer.zero_grad()
            loss, nll_loss, nce_loss, entropy, nce_state, nce_action = model.loss(batch, epoch*8)

            losses.append(loss.item())
            nll_losses.append(nll_loss.item())
            nce_losses.append(nce_loss.item())
            entropies.append(entropy)
            nce_states.append(nce_state.item())
            nce_actions.append(nce_action.item())

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_nll_loss += nll_loss.item()
            total_nce_loss += nce_loss
            total_state_nce += nce_state
            total_action_nce += nce_action
  
        tqdm.write(f"Epoch: {epoch} Loss: {total_loss/len(dataloader)}, NLL Loss: {total_nll_loss/len(dataloader)}, NCE Loss: {total_nce_loss/len(dataloader)}, Entropy: {sum(entropies[-len(dataloader):])/len(dataloader)}")
        
    torch.save(model.state_dict(), "./logs/" + args.model + "/stable.ckpt")
    print("Model saved")

    
    plt.plot(losses, color='blue', label='losses')
    plt.plot(nll_losses, color='red', label='nll_losses')
    plt.plot(nce_losses, color='green', label='nce_losses')
    #plt.plot(nce_states, color='yellow', label='nce_states')
    #plt.plot(nce_actions, color='orange', label='nce_actions')

    plt.xlabel('Epochs')  # or whatever x-axis represents
    plt.ylabel('Loss Value')
    plt.title('Comparison of Loss Metrics')
    plt.legend()
    plt.show()

    plt.plot(entropies)
    plt.show()
    