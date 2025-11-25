import argparse
import time
import datetime

import torch
import numpy as np
import yaml

import environment
import models
import utils

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def find_primitives(model, r, folder_name):
    print(">>> Distilling with Optimization")
    
    a_init = torch.load("data/"+folder_name+"/action.pt").unsqueeze(1)
    print(a_init.shape)

    o_target = torch.zeros((1, 2**r, r))

    a_init = a_init.repeat(1, 2**r, 1)

    for i in range(2**r):
        binary = format(i, '0'+str(r)+'b')
        for j in  range(r):
            o_target[0, i, j] = int(binary[j])
            
    primitives = utils.distill_ST_primitives(model, a_init, o_target, lr=0.001, n_iters=5000, threshold=0.000001)
    
    model = model.cuda()
    
    for i in list(primitives.keys()):
        print(i," -> ", (model.encode_action(primitives[i][1].to("cuda").unsqueeze(0)).detach().cpu().numpy()>=0).astype(int))
    return primitives


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Simulate actions")
    parser.add_argument("-o", "--output", type=str, default="dump", help="Output directory under data")
    parser.add_argument("-n", "--n-interaction", type=int, default=100, help="Number of interactions")
    parser.add_argument("-m", "--model_folder", type=str, help="Model folder")
    parser.add_argument("-d", "--data_folder", type=str, help="Data folder")
    parser.add_argument("-g", "--gui", action="store_true", help="Run with GUI")
    parser.add_argument("-p", "--primitives", type=str, help="Primitives folder")
    parser.add_argument("-f", "--mode", type=str, default="last", help="Run with GUI")
    
    args = parser.parse_args()

    env = environment.BlocksWorld(gui=args.gui, min_objects=1, max_objects=1)
    
    
    if args.model_folder is not None:
        strategy = "model"

        config_file = "logs/"+args.model_folder+"/config.yaml"
        config = yaml.safe_load(open(config_file, "r"))

        if args.model_folder == "random1" or args.model_folder == "active1":
            model = models.SIU_Deepsym(config["state_dim"], 
                                              config["action_dim"], 
                                              config["hidden_dim"], 
                                              config["output_dim"], 
                                              config["state_latent_dim"],
                                              config["action_latent_dim"],
                                              deterministic=False, 
                                              activation=config["activation"]
                                            )
        else:   
            model = models.EffectPredictorDeepSym(config["state_dim"], 
                                                config["action_dim"], 
                                                config["hidden_dim"], 
                                                config["output_dim"], 
                                                config["state_latent_dim"],
                                                config["action_latent_dim"],
                                                deterministic=True, 
                                                activation=config["activation"]
                                                )
        
        model.load_state_dict(torch.load("logs/"+args.model_folder+"/" + args.mode + ".ckpt"))
        
        model = model.to("cuda")
        model.eval()
        
        #model.validate_emb()

        primitives = find_primitives(model, config["action_latent_dim"], args.data_folder)

        primitives_dict = {}
        for key, value in primitives.items():
            # Convert the tensor to a list of native Python floats
            primitives_dict[str(key)] = [float(x) for x in value[1].detach().cpu().numpy()]

        with open("./logs/" + args.model_folder + "/primitives.yaml", "w") as file:
            yaml.safe_dump(primitives_dict, file)

        state_emb = model.encode_state(torch.tensor(env.state(),dtype=torch.float32).to("cuda"))[0]

        print(state_emb)

        print(primitives)

    elif args.primitives is not None:
        strategy = "primitives"

        with open("./logs/"+args.primitives+"/primitives.yaml", "r") as file:
            primitives = yaml.safe_load(file)
        
        for key, value in primitives.items():
            primitives[key] = np.array(value, dtype=np.float32).reshape(3, 4)
        

    else:
        strategy = "random"


    # allocate memory for interaction data
    state_arr  = torch.zeros(args.n_interaction, 4, dtype=torch.float)
    action_arr = torch.zeros(args.n_interaction, 12, dtype=torch.float)
    effect_arr = torch.zeros(args.n_interaction, 3,  dtype=torch.float)
    elem = 0
    double_object_data = []
    # EXPLORATION
    it = 0
    start = time.time()
    while it < args.n_interaction:
        
        if it % 10 == 0:
            # reset to the initial pose before the next interaction
            env.init_agent_pose(t=1)

            env.delete_objects()
            position1, position2 = env.init_random_objects(epoch=None, 
                                                           eval=False, 
                                                           two_object_plan=False, 
                                                           object_idx=2, 
                                                           object_idx2=3, 
                                                           obj_pose=None, 
                                                           obj_pose2=None)


            env._step(240)
            #print(env.get_snapshot()[:,:3])
            n_obj = env.num_objects

        # get the state
        x = torch.tensor(env.state(), dtype=torch.float)

        # select the action with the given strategy
        if strategy == "model":

            anchor, _ = env.get_random_action()
            
            a_p = list(primitives.values())[it][1].numpy().reshape(3, 4)

        elif strategy == "primitives":
            anchor = np.random.randint(0, 2)
            #anchor = 0
            primitives2 = list(primitives.values())
            #primitives2 = [primitives['101'], primitives['110'], primitives['000']]
            a_p = list(primitives2)[it]

        else:
            anchor, a_p = env.get_random_action()

        # env step
        delta, x_next = env.step(anchor, a_p, sleep=False, screenshot=True)
        
        #single_object_data.append([3, position1.tolist(), env.get_snapshot()[:,:3][0].tolist()])
        
        if (delta[anchor]).sum() >= 0.08:
            action_arr[elem] = torch.tensor(a_p.flatten(), dtype=torch.float)
            state_arr[elem] = x[anchor]
            effect_arr[elem] = torch.tensor(delta[anchor], dtype=torch.float)
            elem += 1
            print(elem, "\n")

        #double_object_data.append([[2, 3], [position1.tolist(), position2.tolist()], env.get_snapshot()[:,:3].tolist()])
        it += 1
        end = time.time()
        sample_per_sec = it/(end-start)
        eta = datetime.timedelta(seconds=(args.n_interaction-it)/sample_per_sec)
        print(f"{it}/{args.n_interaction}, "
              f"{sample_per_sec:.2f} samples/sec, ETA={str(eta).split('.', 2)[0]}", end="\r")
        if it%100 == 0:
            print(">>> Saving the data")
            torch.save(state_arr[:elem], "data/"+args.data_folder+"/state.pt")
            torch.save(action_arr[:elem], "data/"+args.data_folder+"/action.pt")
            torch.save(effect_arr[:elem], "data/"+args.data_folder+"/effect.pt")
    print(">>> Saving the data")
    torch.save(state_arr[:elem], "data/"+args.data_folder+"/state.pt")
    torch.save(action_arr[:elem], "data/"+args.data_folder+"/action.pt")
    torch.save(effect_arr[:elem], "data/"+args.data_folder+"/effect.pt")
    '''print(">>> Saving the data")
    print(double_object_data)
    with open("double_object_test.yaml", "a") as file:
        yaml.safe_dump(double_object_data, file)'''