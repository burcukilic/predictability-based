import torch, gc
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm
from models import EffectPredictorDeepSym
import numpy as np
import argparse
import os
import environment
import yaml
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

def get_random_action(x):
    

    a_p = np.zeros((3, 3))
    
    a_p[0] = np.random.uniform((-x, -x, -x/2), (x, x, x/2))
    a_p[1] = np.random.uniform((-x, -x, -x/2), (x, x, x/2))
    a_p[2] = np.random.uniform((-x, -x, -x), (x, x, x))
    
    gripper = np.random.randint(0, 2, (3, 1))

    a_p = np.concatenate([a_p, gripper], axis=-1)
    return a_p

class ForwardBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.effects = []

    def store(self, s, a, e):
        self.states.append(s)
        self.actions.append(a)
        self.effects.append(e)

    def clear(self):
        self.states = []
        self.actions = []
        self.effects = []

    def __len__(self):
        return len(self.states)

epoch = 0
def update_model(model, optimizer, buffer):
    model.train()
    device = next(model.parameters()).device

    states_t = torch.cat(buffer.states, dim=0).to(device)    # shape (B, n_obj, state_dim)
    actions_t = torch.cat(buffer.actions, dim=0).to(device)   # shape (B, action_dim) or (B, 1, action_dim)
    effects_t = torch.cat(buffer.effects, dim=0).to(device)    # shape (B, n_obj, effect_dim)

    
    global epoch
    

    update_epochs = 10

    for _ in range(update_epochs):
        epoch += 1
        batch = (states_t, actions_t, effects_t)

        optimizer.zero_grad()

        loss, nll_loss, nce_loss, entropy, nce_state, nce_action = model.loss(batch, epoch)

        loss.mean().backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()


    return loss.mean().item(), nll_loss.mean().item(), nce_loss.mean().item(), nce_state.mean().item(), nce_action.mean().item(), entropy

def select_action_with_diverse_embedding(state, model, device, episode, num_candidates=10):
    
    candidate_actions = np.array([get_random_action(episode).reshape((12,)) for _ in range(num_candidates)])
    candidate_actions_tensor = torch.tensor(candidate_actions, dtype=torch.float).to(device)
    
    state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
    
    embeddings = []
    for candidate in candidate_actions_tensor:
        candidate = candidate.unsqueeze(0)
        _, _, action_emb = model.forward(state_tensor, candidate)
        embeddings.append(action_emb.squeeze(0).round())
    
    embeddings = torch.stack(embeddings)
    
    distances = torch.cdist(embeddings, embeddings, p=2)
    
    # For each candidate, compute the average distance to all other candidates
    avg_distances = distances.sum(dim=1) / (num_candidates - 1)
    
    best_idx = torch.argmax(avg_distances).item()
    return candidate_actions[best_idx], avg_distances[best_idx].item()

def select_action_with_entropy(state, model, device, episode, num_candidates=10):
    model.eval()
    with torch.no_grad():
        
        candidate_actions = np.array([
            get_random_action(0.05).reshape((12,))
            for _ in range(num_candidates)
        ])
        candidate_actions_tensor = torch.tensor(candidate_actions, dtype=torch.float).to(device)

        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).repeat(num_candidates, 1).to(device)

        # Forward pass all candidate actions at once
        predicted_dist, _, _ = model.forward(state_tensor, candidate_actions_tensor)

        predicted_effect = predicted_dist.loc

        effect_norms = torch.norm(predicted_effect.abs(), dim=1)
        
        valid_mask = effect_norms >= 1e-1
        
        # Compute entropies simultaneously
        entropies = predicted_dist.entropy().mean(dim=1)
        
        if valid_mask.sum() > 0:
            # Filter the entropies and candidate indices by the valid mask
            entropies_valid = entropies[valid_mask]
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            best_valid_idx = torch.argmax(entropies_valid).item()
            best_idx = valid_indices[best_valid_idx].item()
        else:
            # If no candidate meets the threshold, select from all candidates.
            tqdm.write("No valid candidates found.")
            best_idx = torch.argmax(entropies).item()

        if episode % 1000 == 0:
            
            tqdm.write(f"Max & Min Entropy: {torch.max(entropies).item()}, {torch.min(entropies).item()}")
        

        return candidate_actions[best_idx], entropies[best_idx].item()  


def explore(env, 
            episodes, 
            output_folder, 
            data_folder, 
            model_batch_size=512,
            save_steps=512):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "state_dim": 4,
        "action_dim": 12,
        "hidden_dim": 128,
        "output_dim": 3,
        "state_latent_dim": 2,
        "action_latent_dim": 3,
        "activation": "sigmoid",
    }

    model = EffectPredictorDeepSym(**config).to(device)
    
    config["lr"] = 1e-4
    with open("./logs/"+output_folder+"/config.yaml", "w") as file:
        yaml.dump(config, file)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    buffer = ForwardBuffer()

    state_arr  = torch.zeros(episodes, config["state_dim"], dtype=torch.float)
    action_arr = torch.zeros(episodes, config["action_dim"], dtype=torch.float)
    effect_arr = torch.zeros(episodes, config["output_dim"],  dtype=torch.float)

    losses = []
    entropies = []
    elem = 0

    for episode in tqdm(range(episodes)):
        
        if episode > 0 and episode%save_steps == 0:
            
            model.validate_emb()
            model.validate_actions()

            torch.save(model.state_dict(), f"logs/{output_folder}/{episode}.ckpt")
            
            save_folder = os.path.join("data", data_folder)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            torch.save(state_arr, os.path.join(save_folder, f"state.pt"))
            torch.save(action_arr, os.path.join(save_folder, f"action.pt"))
            torch.save(effect_arr, os.path.join(save_folder, f"effect.pt"))

            plt.plot(losses)
            plt.savefig(f"logs/{output_folder}/losses.png")
            plt.close()

            plt.plot(entropies)
            plt.savefig(f"logs/{output_folder}/entropies.png")
            plt.close()

            gc.collect()            
            torch.cuda.empty_cache()  
            
        if episode%1 == 0:
            # reset env 
            env.init_agent_pose(t=1)

            env.delete_objects()
            env.init_random_objects(epoch=None)
            env._step(240)
            n_obj = env.num_objects

            anchor = np.random.randint(0, n_obj)
            state = torch.tensor(env.state(), dtype=torch.float)[anchor].to('cuda')
            
            action_np, entropy = select_action_with_entropy(state, model, device, episode, num_candidates=2048)
            
            action_tensor = torch.tensor(action_np, dtype=torch.float).unsqueeze(0).to(device)

        # Environment Step
        a_p = np.zeros((n_obj, 12))
        a_p[anchor] = action_np

        
        delta, x_next = env.step(anchor, a_p, sleep = False)
        
        s = state.unsqueeze(0).to("cuda")
        a = action_tensor.to("cuda")
        e = torch.tensor(delta, dtype=torch.float)[anchor].unsqueeze(0).to("cuda")
        
        buffer.store(s, a, e)

        state_arr[episode] = s
        action_arr[episode] = a
        effect_arr[episode] = e
        
        if (episode+1) % model_batch_size == 0:
            
            batch_losses, nll_losses, nce_losses, nce_states, nce_actions, entropy = update_model(model, optimizer, buffer)
            tqdm.write(f"Episode: {episode+1}, Loss: {batch_losses}, NCE Loss: {nce_losses}, NLL Loss: {nll_losses}, NCE State: {nce_states}, NCE Action: {nce_actions}, Entropy: {entropy}")
            
            entropies.append(entropy)

            losses.append(batch_losses)
            buffer.clear()
            
    torch.save(model.state_dict(), f"logs/{output_folder}/last.ckpt")
            
    save_folder = os.path.join("data", data_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.save(state_arr, os.path.join(save_folder, f"state.pt"))
    torch.save(action_arr, os.path.join(save_folder, f"action.pt"))
    torch.save(effect_arr, os.path.join(save_folder, f"effect.pt"))
    
    plt.plot(losses)
    plt.savefig(f"logs/{output_folder}/losses.png")
    plt.close()

    plt.plot(entropies)
    plt.savefig(f"logs/{output_folder}/entropies.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Explore the environment")
    parser.add_argument("-o", "--output", type=str, default="dump", help="Output directory")
    parser.add_argument("-e", "--episodes", type=int, default=200, help="Number of epochs")
    parser.add_argument("-jid", "--data-jid", type=str, help="Job ID")
    parser.add_argument("-g", "--gui", action="store_true", help="Run with GUI")
    args = parser.parse_args()

    
    env = environment.BlocksWorld(gui=args.gui, min_objects=1, max_objects=1)
    
    explore(env=env,
            episodes=args.episodes,
            output_folder=args.output,
            data_folder=args.data_jid)

    