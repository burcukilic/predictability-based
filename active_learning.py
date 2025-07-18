import torch, gc
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm
from models import SIU_Deepsym
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
    
    a_p[0] = np.random.uniform((-x, -x, -x), (x, x, x))
    a_p[1] = np.random.uniform((-x, -x, -x), (x, x, x))
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

    states_t = torch.cat(buffer.states, dim=0).to(device)    
    actions_t = torch.cat(buffer.actions, dim=0).to(device) 
    effects_t = torch.cat(buffer.effects, dim=0).to(device)  

    global epoch
    
    update_epochs = 10

    for _ in range(update_epochs):
        epoch += 1
        batch = (states_t, actions_t, effects_t)

        optimizer.zero_grad()

        loss = model.loss(batch, epoch)

        loss.mean().backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()


    return loss.mean().item()


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
        "activation": "ST",
    }

    model = SIU_Deepsym(**config).to(device)
    
    config["lr"] = 1e-5
    with open("./logs/"+output_folder+"/config.yaml", "w") as file:
        yaml.dump(config, file)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    buffer = ForwardBuffer()

    # allocate memory for interaction data
    state_arr  = torch.zeros(episodes, config["state_dim"], dtype=torch.float)
    action_arr = torch.zeros(episodes, config["action_dim"], dtype=torch.float)
    effect_arr = torch.zeros(episodes, config["output_dim"],  dtype=torch.float)

    losses = []
    actions_list = []
    errors_list  = []

    train_turn = 0

    for episode in tqdm(range(episodes)):
        model.eval()

        if episode > 0 and episode%save_steps == 0:

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

            gc.collect()            
            torch.cuda.empty_cache()
            

        if episode%1 == 0:
            # reset env 
            env.init_agent_pose(t=1)
            
            env.delete_objects()
            env.init_random_objects(epoch=None, eval=False)
            env._step(240)
            n_obj = env.num_objects

            anchor = np.random.randint(0, n_obj)
            state = torch.tensor(env.state(), dtype=torch.float)[anchor].to('cuda')
            
            action_np = get_random_action(1.0).reshape((12))
            action_tensor = torch.tensor(action_np, dtype=torch.float).unsqueeze(0).to(device)
            if episode%(100+100) < 100:
                if episode%5==0:
                    
                    action_np = get_random_action(0.1).reshape((12))
                    action_tensor = torch.tensor(action_np, dtype=torch.float).unsqueeze(0).to(device)
            else:
                if episode % (100+100) == 100:

                    actions_list = actions_list[::5]
        
                    errors_list = [np.mean(errors_list[i:i+5]) for i in range(len(errors_list)-4)]
                    errors_list = errors_list[::5]
                    
                    actions_array = np.array(actions_list)
                    errors_array = np.array(errors_list)
                    
                    top_5_indices = np.argsort(errors_array)[-5:]

                    best_5_actions = actions_array[top_5_indices]
                    best_5_errors = errors_array[top_5_indices]
                    tqdm.write(str(best_5_errors))

                    actions_list = []
                    errors_list = []
                action_np = best_5_actions[np.random.randint(0, best_5_actions.shape[0])] + np.random.randn(12)*0.005
                
        # Environment Step
        a_p = np.zeros((n_obj, 12))
        a_p[anchor] = action_np

        
        delta, x_next = env.step(anchor, a_p, sleep = False)
        
        s = state.unsqueeze(0).to("cuda")
        a = action_tensor.to("cuda")
        e = torch.tensor(delta, dtype=torch.float)[anchor].unsqueeze(0).to("cuda")
        
        buffer.store(s, a, e)

        state_arr[episode]  = s
        action_arr[episode] = a
        effect_arr[episode] = e
        
        
        forward_loss = model.loss((s, a, e))

        if (episode + 1) % (100 + 100) == 0:
            train_turn = (train_turn + 1)%3
            
        if (episode + 1) % 200 == 0:
            
            update_model(model, optimizer, buffer)
            buffer.clear()

        success = torch.norm(e[0][anchor]).item() >= 0.0
        if success:
            if episode%(100+100) < 100:
                actions_list.append(action_tensor.squeeze(0).detach().cpu().numpy())
                
                reward = 0*forward_loss.mean().item() + 200*torch.norm(e[0][anchor]).cpu().numpy()
                
                errors_list.append(reward)
                
            else:
                
                reward = 0*forward_loss.mean().item() + 200*torch.norm(e[0][anchor]).cpu().numpy()

        losses.append(forward_loss.mean().item())
            
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
