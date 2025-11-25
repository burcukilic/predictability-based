import torch
import argparse
import yaml
from collections import deque
import numpy as np
import environment

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

from scipy.stats import norm

def bfs_effect_planner(goal_effect, actions, objects, DeepSym_predictor, threshold=0.05, max_depth=10, granularity=2):

    initial_effect = np.zeros_like(goal_effect)

    queue = deque()
    queue.append((initial_effect, [], [], []))  # effects, actions, means, stds

    visited = set()
    visited.add(tuple(np.round(initial_effect, granularity)))

    while queue:
        cumulative_effect, sequence, means_list, stds_list = queue.popleft()

        if np.linalg.norm(cumulative_effect - goal_effect) <= threshold:
            # Compute cumulative mean and std
            cumulative_mean = np.sum(means_list, axis=0)
            cumulative_variance = np.sum([std**2 for std in stds_list], axis=0)
            cumulative_std = np.sqrt(cumulative_variance)

            # Calculate probability of being within threshold of goal
            lower_bound = goal_effect - threshold
            upper_bound = goal_effect + threshold

            probs = norm.cdf(upper_bound, cumulative_mean, cumulative_std) - norm.cdf(lower_bound, cumulative_mean, cumulative_std)
            probability_of_reaching_goal = np.prod(probs)
            
            print(">>> Success")
            print(cumulative_effect)
            
            return sequence, probability_of_reaching_goal

        if len(sequence) >= max_depth:
            continue

        for action in actions:
            for obj in objects:
                predicted_effect, mean, std = DeepSym_predictor(action, obj)
                predicted_effect_np = predicted_effect.detach().cpu().numpy()
                mean_np = mean.detach().cpu().numpy()
                std_np = std.detach().cpu().numpy()

                next_cumulative_effect = cumulative_effect + predicted_effect_np

                discretized_effect = tuple(np.round(next_cumulative_effect, granularity).flatten())

                if discretized_effect in visited:
                    continue

                visited.add(discretized_effect)

                queue.append((
                    next_cumulative_effect,
                    sequence + [(action, obj)],
                    means_list + [mean_np],
                    stds_list + [std_np]
                ))

    print(">>> No valid sequence found.")
    return None, 0.0


def bfs_execute_planner(env, obj_idx, obj_pose, goal_state, actions, objects, action_parameters, obj_idx2=None, obj_pose2=None, threshold=0.05, max_depth=5):
    env.init_agent_pose(t=1)
    env.delete_objects()
    if obj_idx2 is not None:
        
        env.init_random_objects(eval=True, epoch=None, two_object_plan=True, object_idx=obj_idx, obj_pose=obj_pose, object_idx2=obj_idx2, obj_pose2=obj_pose2)
    else:
        env.init_random_objects(eval=True, epoch=None, two_object_plan=False, object_idx=obj_idx, obj_pose=obj_pose)
    env._step(240)

    initial_state = env.get_snapshot()[:,:3]
    print(initial_state)
    queue = deque()
    queue.append((initial_state, []))  # state, actions, means

    visited = set()
    visited.add(tuple(np.round(initial_state.flatten(), 2).tolist()))

    while queue:
        
        cumulative_state, sequence = queue.popleft()
        #print(len(sequence))
        # reset env 
        env.init_agent_pose(t=1)
        env.delete_objects()
        if obj_idx2 is not None:
            env.init_random_objects(eval=True, epoch=None, two_object_plan=True, object_idx=obj_idx, obj_pose=obj_pose, object_idx2=obj_idx2, obj_pose2=obj_pose2)
        else:
            env.init_random_objects(eval=True, epoch=None, two_object_plan=False, object_idx=obj_idx, obj_pose=obj_pose)
        env._step(240)
        
        for action, obj in sequence:
            env.step(obj, action_parameters[action])
        
        if np.linalg.norm(cumulative_state - goal_state) <= threshold:
            print(" >>> Success")
            print(" ", cumulative_state)
            return sequence

        if len(sequence) >= max_depth:
            continue

        for action in actions:
            
            for obj in objects:
                delta, ns = env.step(obj, action_parameters[action])
                next_state = env.get_snapshot()[:,:3]
                next_cumulative_state = 0*cumulative_state + next_state

                discretized_state = tuple(np.round(next_cumulative_state, 2).flatten())

                if discretized_state in visited:
                    
                    continue

                visited.add(discretized_state)

                queue.append((
                    next_cumulative_state,
                    sequence + [(action, obj)]
                ))

    return None
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plan the sequence")
    parser.add_argument("-o", "--output", type=str, default="dump", help="Output directory")
    parser.add_argument("-m", "--model", type=str, help="Model folder name")
    parser.add_argument("-jid", "--data-jid", type=str, help="Data Folder")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model config
    config_file = "./logs/" + args.model + "/config.yaml"

    config = yaml.safe_load(open(config_file, "r"))

    env = environment.BlocksWorld(gui=False, min_objects=1, max_objects=1)

    state_dim = config["state_dim"]
    action_dim = config["action_dim"]
    hidden_dim = config["hidden_dim"]
    output_dim = config["output_dim"]
    state_latent_dim = config["state_latent_dim"]
    action_latent_dim = config["action_latent_dim"]
    activation = config["activation"]

    with open("./logs/"+args.model+"/primitives.yaml", "r") as file:
            primitives = yaml.safe_load(file)
        
    for key, value in primitives.items():
        primitives[key] = np.array(value, dtype=np.float32).reshape(3, 4)

    actions = list(str(i) for i in list(primitives.keys()))
  
    mode = "double_object"
    if mode == "single_object":
        with open("single_object_test.yaml", "r") as file:
            single_object = yaml.safe_load(file)
        
        success = 0
        for i in range(len(single_object)):
            print(f">>> goal {i} started")
            env.init_agent_pose(t=1)
            env.delete_objects()
            env.init_random_objects(eval=True, epoch=None, two_object_plan=False, object_idx=single_object[i][0], obj_pose=single_object[i][1])
            env._step(240)
            result = bfs_execute_planner(env=env,
                                obj_idx = single_object[i][0],
                                obj_pose = single_object[i][1],
                                goal_state=np.array(single_object[i][2]),
                                actions=actions,
                                objects=[0],
                                action_parameters=primitives,
                                threshold=0.05,
                                max_depth=3)
            if result is not None:
                print(" >>> Sequence: ", result)
                success += 1
                print(success)
            else:
                print(" >>> No valid sequence found.")
        print(success)
    else:
        with open("double_object_test.yaml", "r") as file:
            double_object = yaml.safe_load(file)

        success = 0
        for i in range(len(double_object)):
            print(f">>> goal {i} started")
            env.init_agent_pose(t=1)
            env.delete_objects()
            obj_idx = double_object[i][0][0]
            obj_idx2 = double_object[i][0][1]
            obj_pose = double_object[i][1][0]
            obj_pose2 = double_object[i][1][1]
            goal_state = double_object[i][2]
            anchor = 0 if i < 25 else 1
            env.init_random_objects(eval=True, epoch=None, two_object_plan=True, object_idx=obj_idx, obj_pose=obj_pose, object_idx2=obj_idx2, obj_pose2=obj_pose2)
            env._step(240)
            result = bfs_execute_planner(env=env,
                                #anchor=anchor,
                                obj_idx = obj_idx,
                                obj_idx2 = obj_idx2,
                                obj_pose = obj_pose,
                                obj_pose2 = obj_pose2,
                                goal_state=np.array(goal_state),
                                actions=actions,
                                objects=[0, 1],
                                action_parameters=primitives,
                                threshold=0.15,
                                max_depth=3)
            if result is not None:
                print(" >>> Sequence: ", result)
                success += 1
                print(success)
            else:
                print(" >>> No valid sequence found.")
