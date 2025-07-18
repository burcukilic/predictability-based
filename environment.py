import pybullet
import pybullet_data
import numpy as np
from PIL import Image
import os

import utils
import manipulators

step_i = 0

class GenericEnv:
    def __init__(self, gui=0, seed=None):
        self._p = utils.connect(gui)
        self.gui = gui
        self.obj_ids = []
        self.reset(seed=seed)

    def set_camera(self, camera_distance=1.0, camera_yaw=50, camera_pitch=-35, target_position=[0, 0, 0.5]):
        """
        Adjust the camera position and orientation.
        :param camera_distance: Distance from the target position.
        :param camera_yaw: Yaw angle of the camera.
        :param camera_pitch: Pitch angle of the camera.
        :param target_position: The point to focus on.
        """
        self._p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=target_position
        )
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self._p.resetSimulation()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -9.807)
        self._p.loadURDF("plane.urdf")

        self.env_dict = utils.create_tabletop(self._p)
        self.agent = manipulators.Manipulator(p=self._p, path="ur10e/ur10e.urdf", position=[0., 0., 0.4], ik_idx=30)
        base_constraint = self._p.createConstraint(parentBodyUniqueId=self.env_dict["base"], parentLinkIndex=0,
                                                   childBodyUniqueId=self.agent.id, childLinkIndex=-1,
                                                   jointType=self._p.JOINT_FIXED, jointAxis=(0, 0, 0),
                                                   parentFramePosition=(0, 0, 0),
                                                   childFramePosition=(0.0, 0.0, -0.2),
                                                   childFrameOrientation=(0, 0, 0, 1))
        self._p.changeConstraint(base_constraint, maxForce=10000)
        # force grippers to act in sync
        mimic_constraint = self._p.createConstraint(self.agent.id, 28, self.agent.id, 29,
                                                    jointType=self._p.JOINT_GEAR,
                                                    jointAxis=[1, 0, 0],
                                                    parentFramePosition=[0, 0, 0],
                                                    childFramePosition=[0, 0, 0])
        self._p.changeConstraint(mimic_constraint, gearRatio=-1, erp=0.1, maxForce=50)

        self.set_camera(camera_distance=0.8, camera_yaw=90, camera_pitch=-30, target_position=[0.7, 0.0, 0.4])

    def init_agent_pose(self, t=None, sleep=False, traj=False):
        angles = [-0.294, -1.950, 2.141, -2.062, -1.572, 1.277]
        self.agent.set_joint_position(angles, t=t, sleep=sleep, traj=traj)

    def state_obj_poses(self):
        N_obj = 1
        pose = np.zeros((N_obj, 7), dtype=np.float32)
        for i in range(N_obj):
            position, quaternion = self._p.getBasePositionAndOrientation(self.obj_dict[i])
            pose[i][:3] = position
            pose[i][3:] = quaternion
        return pose

    def get_contact_graph(self):
        N_obj = len(self.obj_dict)
        contact_graph = np.zeros((N_obj, N_obj), dtype=int)
        reverse_obj_dict = {v: k for k, v in self.obj_dict.items()}
        for i in range(N_obj):
            contacts = self._p.getContactPoints()
            for contact in contacts:
                if (contact[1] in reverse_obj_dict) and (contact[2] in reverse_obj_dict):
                    contact_graph[reverse_obj_dict[contact[1]], reverse_obj_dict[contact[2]]] = 1
                    contact_graph[reverse_obj_dict[contact[2]], reverse_obj_dict[contact[1]]] = 1

        for i in range(contact_graph.shape[0]):
            neighbors = np.where(contact_graph[i] == 1)[0]
            for ni in neighbors:
                for nj in neighbors:
                    contact_graph[ni, nj] = 1
            contact_graph[i, i] = 1
        return contact_graph

    def _step(self, count=1):
        for _ in range(count):
            self._p.stepSimulation()

    def __del__(self):
        self._p.disconnect()


class BlocksWorld(GenericEnv):
    def __init__(self, gui=0, seed=None, min_objects=5, max_objects=5):
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.num_objects = None
        
        
        super(BlocksWorld, self).__init__(gui=gui, seed=seed)

    def reset(self, seed=None):
        super(BlocksWorld, self).reset(seed=seed)

        self.obj_dict = {}
        self.obj_size = {}
        self.type_arr = {}
        self.init_agent_pose(t=1)
        #self.init_objects()
        self.init_random_objects()
        self._step(40)
        self.agent.open_gripper(1, sleep=True)

    def delete_objects(self):
        for key in self.obj_dict:
            obj_id = self.obj_dict[key]
            self._p.removeBody(obj_id)
        self.obj_dict = {}
        self.obj_size = {}
        self.type_arr = {}

    def reset_objects(self, epoch=None):
        self.delete_objects()
        self.init_objects(epoch=epoch)
        self._step(240)

    def reset_object_poses(self):
        for key in self.obj_dict:
            x = np.random.uniform(0.5, 1.0)
            y = np.random.uniform(-0.4, 0.4)
            z = np.random.uniform(0.6, 0.65)
            quat = pybullet.getQuaternionFromEuler(np.random.uniform(0, 90, (3,)).tolist())

            self._p.resetBasePositionAndOrientation(self.obj_dict[key], [x, y, z], quat)
        self._step(240)

    def setup_objects(self):
        
        colors = [[0.8,0,0,1], [0,0,0.8,1], [0,0.8,0,1], [0.8,0.8,0,1]]
        
        self.sizes = [
            [0.07,0.07,0.07],
            #[0.05, 0.05, 0.05],
            [0.03, 0.1, 0.02],
            [0.04, 0.05, 0.],
            [0.1, 0.1, 0.04]
        ]
        self.otypes = {}

        offscreen = [0,0,-10]

        for i in range(4):

            obj_type = [self._p.GEOM_BOX, self._p.GEOM_BOX, "hollow", "ring"][i]
            color = colors[i]

            obj_id = utils.create_object(self._p, obj_type, self.sizes[i], offscreen, color=color, mass=0.1)
            
            self._p.resetBasePositionAndOrientation(obj_id, offscreen, [0,0,0,1])
            self.obj_ids.append(obj_id)
            self.otypes[obj_id] = i
      
        print(self.otypes)
        print(self.obj_ids)

    def init_objects(self, epoch=None):

        self.num_objects = 1

        if self.obj_ids == []:
            self.setup_objects()
        # Move all objects out of sight first
        for oid in self.obj_ids:
            self._p.resetBasePositionAndOrientation(oid, [0,0,-10], [0,0,0,1])

        # Choose randomly
        if epoch is not None:
            if epoch < 2049:
                chosen_id = np.random.choice([self.obj_ids[0]])
            elif epoch < 4097:
                chosen_id = np.random.choice([self.obj_ids[0], self.obj_ids[1]], p=[0.2, 0.8])
            elif epoch < 6145:
                chosen_id = np.random.choice([self.obj_ids[0], self.obj_ids[1], self.obj_ids[2]], p=[0.1, 0.1, 0.8])
            elif epoch < 8193:
                chosen_id = np.random.choice([self.obj_ids[0], self.obj_ids[1], self.obj_ids[2], self.obj_ids[3]], p=[0.1, 0.1, 0.1, 0.7])
            else:
                chosen_id = np.random.choice(self.obj_ids)
        else:
            # chosen_id = self.obj_ids[2]
            chosen_id = np.random.choice(self.obj_ids)
        
        position = np.array([0.75, 0.0, 0.50])
        
        self._p.resetBasePositionAndOrientation(chosen_id, position, [0,0,0,1])
        
        self.obj_dict[0] = chosen_id
        self.obj_size[0] = self.sizes[self.otypes[chosen_id]]
        self.type_arr[0] = self.otypes[chosen_id]
        
        return chosen_id
    
    def init_random_objects(self, epoch=None, eval=False, two_object_plan=False, object_idx=None, object_idx2=None, obj_pose=None, obj_pose2=None):
        obj_types = [self._p.GEOM_BOX, self._p.GEOM_BOX, "hollow", "ring"]
        colors = [[0.8, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.8, 1.0],
                  [0.0, 0.8, 0.0, 1.0],
                  [0.8, 0.8, 0.0, 1.0]]

        size_limits = [[(0.068, 0.068, 0.068), (0.072, 0.072, 0.072)], 
                       [(0.028, 0.098, 0.018), (0.032, 0.102, 0.022)], 
                       [(0.038, 0.048, -0.002), (0.042, 0.052, 0.002)],
                       #[(0.03, 0.03, 0.001), (0.032, 0.032, 0.001)],
                        [(0.098, 0.098, 0.038), (0.12, 0.12, 0.042)]]
                        #[(0.16, 0.16, 0.04), (0.14, 0.14, 0.042)]]


        obj_ids = []
        obj_sizes = []
        type_arr = []
        #n_obj = self.max_objects#np.random.randint(self.min_objects, self.max_objects+1)
        if two_object_plan:
            n_obj = 2
        else:
            n_obj = 1
        position = None
        position2 = None
        self.num_objects = n_obj
        for k in range(n_obj):
            if epoch is not None:
                if epoch < 1000:
                    idx = 2
                elif epoch < 2000:
                    idx = 0
                elif epoch < 3000:
                    idx = 0
                elif epoch < 4000:
                    idx = 1
                else:
                    idx = np.random.choice([0, 1, 2], p=[0.5, 0.25, 0.25])
            else:
                
                idx = np.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
            if eval:
                if two_object_plan:
                    idx = [object_idx, object_idx2][k]
                    
                else:
                    if object_idx is not None:
                        idx = object_idx
                    else:
                        idx = 0
                        #print("Please provide object index for single-object evaluation")
                        
                    
            obj_type = obj_types[idx]
            color = colors[idx]

            size = np.random.uniform(size_limits[idx][0], size_limits[idx][1], (3,))
            if eval:
                size = size_limits[idx][0]

            if two_object_plan:
                if obj_pose is not None:
                    position = obj_pose if k == 0 else obj_pose2
                else:
                    position = np.random.uniform((0.25, -0.25, 0.50), (0.75, 0.25, 0.50), (3,)).round(3)
                    position2= np.random.uniform((0.25, -0.25, 0.50), (0.75, 0.25, 0.50), (3,)).round(3)

            elif object_idx is not None:
                if obj_pose is not None:
                    position = obj_pose
                else:
                    #print("obj pos not given")
                    position = np.random.uniform((0.25, -0.25, 0.50), (0.75, 0.25, 0.50), (3,)).round(3)
                    #print("initial pos: ", position)
            else:
                #print("initialized")
                position = np.array([0.75, 0.0, 0.50])
            
            obj_ids.append(utils.create_object(p=self._p, obj_type=obj_type, size=size, position=position,
                                               color=color, mass=0.1))
            obj_sizes.append(size)
            type_arr.append(idx)
            

        for i, (o_id, o_size, o_type) in enumerate(sorted(zip(obj_ids, obj_sizes, type_arr))):
            self.obj_dict[i] = o_id
            self.obj_size[i] = o_size
            self.type_arr[i] = o_type
        
        return position, position2
    
    def state(self):
        # rgb, depth, seg = utils.get_image(p=self._p, eye_position=[0.85, 0.0, 1.85], target_position=[0.8, 0., 0.45],
        #                                   up_vector=[0, 0, 1], height=512, width=512)
        # return rgb[:, :, :3], depth, seg
        #poses = self.state_obj_poses()
        sizes = np.array([self.obj_size[key] for key in self.obj_size])
        types = np.array([[self.type_arr[key]] for key in self.type_arr])
        types = types > 1
        #touch = np.zeros((self.num_objects, 1), dtype=np.float32)
        #contact_obj = self.agent.get_contacted_object()
        '''if contact_obj != -1:
            rev_dict = {v: k for k, v in self.obj_dict.items()}
            print(rev_dict)
            print(contact_obj)
        
            contact_obj = rev_dict[contact_obj]
            touch[contact_obj] = 1'''
        #state = np.concatenate([poses, sizes, types, touch], axis=-1)
        #state = np.concatenate([poses, sizes, types], axis=-1)
        state = np.concatenate([sizes, types], axis=-1)
        
        return state
    
    def get_snapshot(self):
        
        
        poses = self.state_obj_poses()
        
        types = np.array([[self.type_arr[key]] for key in self.type_arr])
        
        state = np.concatenate([poses[:,:3], types], axis=-1)
        
        return state

    def get_random_action(self):
    

        a_p = np.zeros((3, 3))
        
        a_p[0] = np.random.uniform((-0.05, -0.05, -0.05/2), (0.05, 0.05, 0.05/2))
        a_p[1] = np.random.uniform((-0.05, -0.05, -0.05/2), (0.05, 0.05, 0.05/2))
        a_p[2] = np.random.uniform((-0.05, -0.05, -0.05),   (0.05, 0.05, 0.05))
        
        gripper = np.random.randint(0, 2, (3, 1))

        a_p = np.concatenate([a_p, gripper], axis=-1)
        return np.random.randint(0, self.num_objects), a_p


    def get_push_action(self):
        anchor_idx = np.random.randint(0, self.num_objects)
        a_p = np.zeros((3, 3))
        a_p[0] = np.array([0.0, -0.3, 0.1])
        a_p[1] = np.array([0.0, -0.1, 0.0])
        a_p[2] = np.array([0.0, 0.2, 0.1])
        gripper = np.array([1, 0, 0]).reshape(3, 1)

        a_p = np.concatenate([a_p, gripper], axis=-1)

        return anchor_idx, a_p

    def get_pull_action(self):
        anchor_idx = np.random.randint(0, self.num_objects)
        start_position = np.array([0.0, 0.4, -0.1])
        end_position = np.array([0.0, -0.4, -0.1])

        noise_std = 0.06  # Standard deviation of the noise
        noise_start = np.random.normal(0, noise_std, start_position.shape)
        noise_end = np.random.normal(0, noise_std, end_position.shape)

        # Add noise to positions
        start_position = start_position + noise_start
        end_position = end_position + noise_end

        trajectory = np.linspace(start_position, end_position, 3)

        gripper = np.zeros((3, 1))

        a_p = np.hstack([trajectory, gripper])

        return anchor_idx, a_p
    
    def pick_place_action(self):
        anchor_idx = np.random.randint(0, self.num_objects)
        a_p = np.zeros((3, 3))
        a_p[0] = np.array([0.0, 0.0, 0.0])
        a_p[1] = np.array([0.0, 0.0, 0.0])
        a_p[2] = np.array([0.0, 0.2, 0.6])
        gripper = np.array([1, 0, 0]).reshape(3, 1)

        a_p = np.concatenate([a_p, gripper], axis=-1)

        return anchor_idx, a_p

    def step(self, anchor_idx, action_params, sleep=False, screenshot = False):
        """
        Execute the action given the action parameters that defines
        three via points.

        Parameters
        ----------
        anchor_idx : int
            Index of the object to interact with.
        action_params : np.ndarray
            (n x 4) array of n via points. Each point is [x, y, z, gripper].
        sleep : bool
            Whether to sleep the simulation or not.
        """
        anchor_point = self.state_obj_poses()[anchor_idx, :3]
        # use a fixed orientation at the moment
        quat = self._p.getQuaternionFromEuler([np.pi, 0, np.pi/2])

        action_params = action_params.copy()
        action_params[:, :3] = action_params[:, :3] + anchor_point

        # clip positions to limits
        '''action_params[:, 0] = np.clip(action_params[:, 0], 0.5, 1.0)
        action_params[:, 1] = np.clip(action_params[:, 1], -0.5, 0.5)
        action_params[:, 2] = np.clip(action_params[:, 2], 0.42, 1.0)'''

        if screenshot:
            point1 = utils.create_object_image(p=self._p, obj_type=self._p.GEOM_SPHERE,
                                               size=(0.01, 0.01, 0.01),
                                               position=action_params[0, :3],
                                               color=[0, 0, 1, 1])
            point2 = utils.create_object_image(p=self._p, obj_type=self._p.GEOM_SPHERE,
                                               size=(0.01, 0.01, 0.01),
                                               position=action_params[1, :3],
                                               color=[0, 1, 0, 1])
            point3 = utils.create_object_image(p=self._p, obj_type=self._p.GEOM_SPHERE,
                                               size=(0.01, 0.01, 0.01),
                                               position=action_params[2, :3],
                                               color=[1, 0, 0, 1])
            debug_items = [point1, point2, point3]

        # go to the initial up position
        self.agent.move_in_cartesian([action_params[0, 0],
                                      action_params[0, 1],
                                      action_params[0, 2]+0.2],
                                     quat, speed=4.0, sleep=sleep)
        self.agent._waitsleep(0.5)

        # estimate initial object positions (fixed for now)
        p_init = self.state_obj_poses()[:, :3]

        global step_i
        

        if screenshot:
            # Capture screenshot
            width, height, rgb_img, _, _ = self._p.getCameraImage(
                width=640,
                height=480,
                #viewMatrix=view_matrix,
                #projectionMatrix=proj_matrix,
                renderer=self._p.ER_BULLET_HARDWARE_OPENGL,  # High-quality rendering
                lightDirection=[1, 1, 1],  # Adjust light source direction for shadows
                shadow=1  # Enable shadows
            )
            img = np.array(rgb_img, dtype=np.uint8).reshape((height, width, 4))  # RGBA format
            img = Image.fromarray(img[:, :, :3])  # Remove alpha channel
            
            # Save the screenshot
            img.save(f"screenshots/step_{step_i}.png")
            step_i += 1

        # execute the action
        for i, action in enumerate(action_params):
            if action[3] > 0.5:
                self.agent.open_gripper(t=0.1)
            else:
                self.agent.close_gripper(t=0.1)
            self.agent.move_in_cartesian(action[:3], quat, speed=10.0, sleep=sleep)
            self.agent._waitsleep(0.1)
            
            
            if screenshot:
                # Capture screenshot
                width, height, rgb_img, _, _ = self._p.getCameraImage(
                    width=640,
                    height=480,
                    #viewMatrix=view_matrix,
                    #projectionMatrix=proj_matrix,
                    renderer=self._p.ER_BULLET_HARDWARE_OPENGL,  # High-quality rendering
                    lightDirection=[1, 1, 1],  # Adjust light source direction for shadows
                    shadow=1  # Enable shadows
                )
                img = np.array(rgb_img, dtype=np.uint8).reshape((height, width, 4))  # RGBA format
                img = Image.fromarray(img[:, :, :3])  # Remove alpha channel
                
                
                # Save the screenshot
                img.save(f"screenshots/step_{step_i}.png")
                step_i += 1

        
        self.agent.move_in_cartesian([action_params[-1, 0],
                                    action_params[-1, 1],
                                    action_params[-1, 2]+0.4],
                                    quat, speed=4.0, sleep=sleep)
        self.agent.open_gripper(t=0.1)

        if screenshot:
            
            # Capture screenshot
            width, height, rgb_img, _, _ = self._p.getCameraImage(
                width=640,
                height=480,
                #viewMatrix=view_matrix,
                #projectionMatrix=proj_matrix,
                renderer=self._p.ER_BULLET_HARDWARE_OPENGL,  # High-quality rendering
                lightDirection=[1, 1, 1],  # Adjust light source direction for shadows
                shadow=1  # Enable shadows
            )
            img = np.array(rgb_img, dtype=np.uint8).reshape((height, width, 4))  # RGBA format
            img = Image.fromarray(img[:, :, :3])  # Remove alpha channel
            
            
            # Save the screenshot
            img.save(f"screenshots/step_{step_i}.png")
            step_i += 1


        # estimate final object positions (fixed for now)
        p_final = self.state_obj_poses()[:, :3]
        next_state = self.state()

        # move upwards to avoid collision
        
        
        # compute the effect
        delta = p_final - p_init

        if screenshot:
            for item in debug_items:
                self._p.removeBody(item)

        return delta, next_state
