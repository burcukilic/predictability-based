import numpy as np
from scipy.spatial.transform import Rotation
import torch

def connect(gui=1):
    import pkgutil
    from pybullet_utils import bullet_client
    if gui:
        p = bullet_client.BulletClient(connection_mode=bullet_client.pybullet.GUI)
    else:
        p = bullet_client.BulletClient(connection_mode=bullet_client.pybullet.DIRECT)
        egl = pkgutil.get_loader("eglRenderer")
        if (egl):
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        else:
            p.loadPlugin("eglRendererPlugin")
    return p

def create_ring(p, radius, thickness, height, position, rotation=[0,0,0], mass=1, color=[1,0,0,1], segments=12):
    angle_step = 2 * np.pi / segments
    half_height = height / 2
    box_length = 2 * radius * np.tan(np.pi / segments)

    collision_shape = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[thickness/2, box_length/2, half_height]
    )

    visual_shape = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[thickness/2, box_length/2, half_height], rgbaColor=color
    )

    linkMasses = [mass / segments] * (segments - 1)
    linkCollisionShapeIndices = [collision_shape] * (segments - 1)
    linkVisualShapeIndices = [visual_shape] * (segments - 1)
    linkPositions = []
    linkOrientations = []
    linkInertialFramePositions = [[0,0,0]]*(segments - 1)
    linkInertialFrameOrientations = [[0,0,0,1]]*(segments - 1)
    linkParentIndices = [0]*(segments - 1)
    linkJointTypes = [p.JOINT_FIXED]*(segments - 1)
    linkJointAxis = [[0,0,0]]*(segments - 1)

    base_theta = 0
    base_position = [
        radius * np.cos(base_theta),
        radius * np.sin(base_theta),
        0
    ]
    base_orientation = p.getQuaternionFromEuler([0, 0, base_theta])

    for i in range(1, segments):
        theta = i * angle_step
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        rel_pos = [x - base_position[0], y - base_position[1], 0]
        quat = p.getQuaternionFromEuler([0, 0, theta])

        linkPositions.append(rel_pos)
        linkOrientations.append(quat)

    obj_id = p.createMultiBody(
        baseMass=mass / segments,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=p.getQuaternionFromEuler(rotation),
        linkMasses=linkMasses,
        linkCollisionShapeIndices=linkCollisionShapeIndices,
        linkVisualShapeIndices=linkVisualShapeIndices,
        linkPositions=linkPositions,
        linkOrientations=linkOrientations,
        linkInertialFramePositions=linkInertialFramePositions,
        linkInertialFrameOrientations=linkInertialFrameOrientations,
        linkParentIndices=linkParentIndices,
        linkJointTypes=linkJointTypes,
        linkJointAxis=linkJointAxis,
    )

    return obj_id



def create_object(p, obj_type, size, position, rotation=[0, 0, 0], mass=1, color=None, with_link=False):
    collisionId = -1
    visualId = -1
    if color == "random":
        color = np.random.rand(3).tolist() + [1]

    if obj_type == p.GEOM_SPHERE:
        collisionId = p.createCollisionShape(shapeType=obj_type, radius=size[0])
        visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], rgbaColor=color)

    elif obj_type in [p.GEOM_CAPSULE, p.GEOM_CYLINDER]:
        collisionId = p.createCollisionShape(shapeType=obj_type, radius=size[0], height=size[1])
        visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], length=size[1], rgbaColor=color)

    elif obj_type == p.GEOM_BOX:
        collisionId = p.createCollisionShape(shapeType=obj_type, halfExtents=size)
        visualId = p.createVisualShape(shapeType=obj_type, halfExtents=size, rgbaColor=color)

    elif obj_type == "hollow":
        height, width = size[:2]
        baseCollisionId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[width, width, 0.002])
        baseVisualId = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[width, width, 0.002], rgbaColor=color)
        edgeCollisionId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[width, 0.002, height])
        edgeVisualId = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[width, 0.002, height], rgbaColor=color)
        obj_id = p.createMultiBody(baseMass=mass/5, baseCollisionShapeIndex=baseCollisionId,
                                   baseVisualShapeIndex=baseVisualId,
                                   basePosition=[position[0], position[1], position[2]-height],
                                   baseOrientation=p.getQuaternionFromEuler(rotation),
                                   linkMasses=[mass/5, mass/5, mass/5, mass/5],
                                   linkCollisionShapeIndices=[edgeCollisionId]*4,
                                   linkVisualShapeIndices=[edgeVisualId]*4,
                                   linkPositions=[
                                       [0, width, height],  # left
                                       [0, -width, height],  # right
                                       [-width, 0, height],  # back
                                       [width, 0, height],  # front
                                   ],
                                   linkOrientations=[
                                     [0, 0, 0, 1],
                                     [0, 0, 0, 1],
                                     p.getQuaternionFromEuler([0, 0, np.pi/2]),
                                     p.getQuaternionFromEuler([0, 0, np.pi/2]),
                                   ],
                                   linkInertialFramePositions=[[0, 0, 0]]*4,
                                   linkInertialFrameOrientations=[[0, 0, 0, 1]]*4,
                                   linkParentIndices=[0, 0, 0, 0],
                                   linkJointTypes=[p.JOINT_FIXED]*4,
                                   linkJointAxis=[[0, 0, 0]]*4)
        return obj_id
    
    elif obj_type == "ring":
        obj_id = create_ring(p, 0.1, 0.02, 0.02, position, rotation, mass, color)
        return obj_id

    elif obj_type == "random":
        obj = "%03d" % np.random.randint(1000)
        obj_id = p.loadURDF(f"random_urdfs/{obj}/{obj}.urdf", basePosition=position,
                            baseOrientation=p.getQuaternionFromEuler(rotation))
        return obj_id

    if with_link:
        obj_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1,
                                   basePosition=position, baseOrientation=p.getQuaternionFromEuler(rotation),
                                   linkMasses=[mass], linkCollisionShapeIndices=[collisionId],
                                   linkVisualShapeIndices=[visualId], linkPositions=[[0, 0, 0]],
                                   linkOrientations=[[0, 0, 0, 1]], linkInertialFramePositions=[[0, 0, 0]],
                                   linkInertialFrameOrientations=[[0, 0, 0, 1]], linkParentIndices=[0],
                                   linkJointTypes=[p.JOINT_FIXED], linkJointAxis=[[0, 0, 0]])
    else:
        obj_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collisionId, baseVisualShapeIndex=visualId,
                                   basePosition=position, baseOrientation=p.getQuaternionFromEuler(rotation))
        p.changeDynamics(obj_id, -1, rollingFriction=0.0005, spinningFriction=0.001)

    return obj_id


def create_arrow(p, from_loc, to_loc, color=[0.0, 1.0, 1.0, 0.75]):
    delta = (np.array(to_loc) - np.array(from_loc))
    length = np.linalg.norm(delta)
    r_x = -np.arctan2(np.linalg.norm([delta[0], delta[1]]), delta[2])
    r_y = 0
    r_z = -np.arctan2(delta[0], delta[1])

    baseVisualId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01,
                                       rgbaColor=[0.0, 0.0, 1.0, 0.75])
    childVisualId = p.createVisualShape(shapeType=p.GEOM_CAPSULE, radius=0.0075, length=length,
                                        rgbaColor=color)
    tipVisualId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01,
                                      rgbaColor=[1.0, 0.0, 1.0, 0.75])
    obj_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=baseVisualId,
                               basePosition=from_loc, baseOrientation=[0., 0., 0., 1], linkMasses=[-1, -1],
                               linkCollisionShapeIndices=[-1, -1], linkVisualShapeIndices=[childVisualId, tipVisualId],
                               linkPositions=[delta/2, delta],
                               linkOrientations=[p.getQuaternionFromEuler([r_x, r_y, r_z]), [0, 0, 0, 1]],
                               linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
                               linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
                               linkParentIndices=[0, 0], linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED],
                               linkJointAxis=[[0, 0, 0], [0, 0, 0]])
    return obj_id


def create_object_image(p, obj_type, size, position, rotation=[0, 0, 0], color="random"):
    visualId = -1
    if color == "random":
        color = np.random.rand(3).tolist() + [0.1]
    else:
        color = color[:3] + [0.55]

    if obj_type == p.GEOM_SPHERE:
        visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], rgbaColor=color)
    elif obj_type in [p.GEOM_CAPSULE, p.GEOM_CYLINDER]:
        visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], length=size[1], rgbaColor=color)
    elif obj_type == p.GEOM_BOX:
        visualId = p.createVisualShape(shapeType=obj_type, halfExtents=size, rgbaColor=color)

    obj_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualId,
                               basePosition=position, baseOrientation=p.getQuaternionFromEuler(rotation))
    return obj_id


def create_tabletop(p):
    objects = {}
    objects["base"] = create_object(p, p.GEOM_BOX, mass=0, size=[0.15, 0.15, 0.2],
                                    position=[0., 0., 0.2], color=[0.5, 0.5, 0.5, 1.0], with_link=True)
    objects["table"] = create_object(p, p.GEOM_BOX, mass=0, size=[0.7, 1, 0.2],
                                     position=[0.9, 0, 0.2], color=[0.9, 0.9, 0.9, 1.0])
    # walls
    objects["wall1"] = create_object(p, p.GEOM_BOX, mass=0, size=[0.7, 0.01, 0.05],
                                     position=[0.9, -1, 0.45], color=[0.4, 0.4, 1.0, 1.0])
    objects["wall2"] = create_object(p, p.GEOM_BOX, mass=0, size=[0.7, 0.01, 0.05],
                                     position=[0.9, 1, 0.45], color=[0.4, 0.4, 1.0, 1.0])
    objects["wall3"] = create_object(p, p.GEOM_BOX, mass=0, size=[0.01, 1, 0.05],
                                     position=[0.2, 0., 0.45], color=[0.4, 0.4, 1.0, 1.0])
    objects["wall4"] = create_object(p, p.GEOM_BOX, mass=0, size=[0.01, 1, 0.05],
                                     position=[1.6, 0., 0.45], color=[0.4, 0.4, 1.0, 1.0])
    return objects


def get_image(p, eye_position, target_position, up_vector, height, width):
    viewMatrix = p.computeViewMatrix(cameraEyePosition=eye_position,
                                     cameraTargetPosition=target_position,
                                     cameraUpVector=up_vector)
    projectionMatrix = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=1.0, farVal=1.50)
    _, _, rgb, depth, seg = p.getCameraImage(height=height, width=width, viewMatrix=viewMatrix,
                                             projectionMatrix=projectionMatrix)
    return rgb, depth, seg


def create_camera(p, position, rotation, static=True):
    baseCollision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
    targetCollision = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=0.005, height=0.01)
    baseVisual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[0, 0, 0, 1])
    targetVisual = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=0.005, length=0.01,
                                       rgbaColor=[0.8, 0.8, 0.8, 1.0])

    # base = create_object(obj_type=p.GEOM_SPHERE, size=0.1, position=position, rotation=rotation)
    # target = create_object(obj_T)
    mass = 0 if static else 0.1
    obj_id = p.createMultiBody(baseMass=mass,
                               baseCollisionShapeIndex=-1,
                               baseVisualShapeIndex=-1,
                               basePosition=position,
                               baseOrientation=p.getQuaternionFromEuler(rotation),
                               linkMasses=[mass, mass],
                               linkCollisionShapeIndices=[baseCollision, targetCollision],
                               linkVisualShapeIndices=[baseVisual, targetVisual],
                               linkPositions=[[0, 0, 0], [0.02, 0, 0]],
                               linkOrientations=[[0, 0, 0, 1], p.getQuaternionFromEuler([0., np.pi/2, 0])],
                               linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
                               linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
                               linkParentIndices=[0, 1],
                               linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED],
                               linkJointAxis=[[0, 0, 0], [0, 0, 0]])

    return obj_id


def get_image_from_cam(p, camera_id, height, width):
    cam_state = p.getLinkStates(camera_id, [0, 1])
    base_pos = cam_state[0][0]
    up_vector = Rotation.from_quat(cam_state[0][1]).as_matrix()[:, -1]
    target_pos = cam_state[1][0]
    target_vec = np.array(target_pos) - np.array(base_pos)
    target_vec = (target_vec / np.linalg.norm(target_vec))
    return get_image(base_pos+target_vec*0.04, base_pos+target_vec, up_vector, height, width)


def get_parameter_count(model):
    total_num = 0
    for param in model.parameters():
        total_num += param.shape.numel()
    return total_num


def print_module(module, name, space):
    L = len(name)
    line = " "*space+"-"*(L+4)
    print(line)
    print(" "*space+"  "+name+"  ")
    print(line)
    module_str = module.__repr__()
    print("\n".join([" "*space+mstr for mstr in module_str.split("\n")]))

#####################################
def distill_ST_primitives(model, a_init, o_target, lr=0.001, n_iters=10000, threshold=0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.requires_grad_(False)

    a_init = a_init.clone().to(device)
    o_target = (o_target).clone().to(device)

    a_init.requires_grad = True
    optim = torch.optim.SGD([a_init], lr=lr) # changed SGD to Adam

    # distill the action primitive with gradient descent
    running_error = 0.0
    for i in range(n_iters):
        
        o = model.encode_action(a_init).round()
        
        error = torch.nn.functional.binary_cross_entropy(o, o_target.repeat(a_init.shape[0], 1, 1), reduction="none").mean(dim=[0, 2]).sum()
        #error = torch.nn.functional.mse_loss(o, o_target.repeat(a_init.shape[0], 1, 1), reduction="none").mean(dim=[0, 2]).sum()
        optim.zero_grad()
        error.backward()
        optim.step()
        #running_error = 0.9 * running_error + 0.1 * error.item()
        running_error = error.item()
        print(f"{i}/{n_iters} error={running_error:.3f}", end="\r")

    # filter actions that are close enough to the target
    error = torch.nn.functional.binary_cross_entropy(o, o_target.repeat(a_init.shape[0], 1, 1), reduction="none").sum(dim=2)
    #error = torch.nn.functional.mse_loss(o, o_target.repeat(a_init.shape[0], 1, 1), reduction="none").mean(dim=2)
    
    elites = error < threshold
    num_elites = elites.sum(dim=0)
    print(num_elites)
    # write the action primitives to a dictionary
    primitives = {}
    o_target = (o_target)
    o_str = binary_tensor_to_str(o_target[0].byte())
    a_init = a_init.detach()

    for i, (n_i, o_i) in enumerate(zip(num_elites, o_str)):
        e_i = elites[:, i]
        if n_i > 0:

            primitives[o_i] = (n_i.item(), a_init[e_i, i].mean(dim=0).cpu())
        #else:
        #    primitives[o_i] = (n_i.item(), None)
    return primitives

def distill_primitives(model, a_init, o_target, lr=0.001, n_iters=10000, threshold=0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.requires_grad_(False)

    a_init = a_init.clone().to(device)
    o_target = (o_target*2-1).clone().to(device)

    a_init.requires_grad = True
    optim = torch.optim.SGD([a_init], lr=lr) # changed SGD to Adam

    # distill the action primitive with gradient descent
    running_error = 0.0
    for i in range(n_iters):
        
        #o = (model.encode_action(a_init)>=0).float().requires_grad_(True)
        b, n, d = a_init.shape  # here, d=12
        action_flat = a_init.view(b * n, d)  # shape [batch_size * 8, 12]
        o = model.encode_action(action_flat)
        o = o.view(b, n, -1)
        #error = torch.nn.functional.binary_cross_entropy(o, o_target.repeat(a_init.shape[0], 1, 1), reduction="none").mean(dim=[0, 2]).sum()
        error = torch.nn.functional.mse_loss(o, o_target.repeat(a_init.shape[0], 1, 1), reduction="none").mean(dim=[0, 2]).sum()
        optim.zero_grad()
        error.backward()
        optim.step()
        #running_error = 0.9 * running_error + 0.1 * error.item()
        running_error = error.item()
        print(f"{i}/{n_iters} error={running_error:.3f}", end="\r")

    # filter actions that are close enough to the target
    #error = torch.nn.functional.binary_cross_entropy(o, o_target.repeat(a_init.shape[0], 1, 1), reduction="none").sum(dim=2)
    error = torch.nn.functional.mse_loss(o, o_target.repeat(a_init.shape[0], 1, 1), reduction="none").mean(dim=2)
    
    elites = error < threshold
    num_elites = elites.sum(dim=0)
    print(num_elites)
    # write the action primitives to a dictionary
    primitives = {}
    o_target = (o_target+1)//2
    o_str = binary_tensor_to_str(o_target[0].byte())
    a_init = a_init.detach()

    for i, (n_i, o_i) in enumerate(zip(num_elites, o_str)):
        e_i = elites[:, i]
        if n_i > 0:
            min_idx = torch.argmin(error[e_i, i])

            best_action = a_init[e_i, i][min_idx]

            primitives[o_i] = (n_i.item(), best_action.cpu())

            #primitives[o_i] = (n_i.item(), a_init[e_i, i].mean(dim=0).cpu())
        #else:
        #    primitives[o_i] = (n_i.item(), None)
    return primitives

def decimal_to_binary(number_list, length=None):
    binaries = [format(x, "0"+str(length)+"b") for x in number_list]
    return binaries


def binary_to_decimal(x):
    N, D = x.shape
    dec_tensor = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    for i in reversed(range(D)):
        multiplier = 2**i
        dec_tensor += multiplier * x[:, D-i-1].int()
    return dec_tensor


def binary_tensor_to_str(x):
    return ["".join([str(x_ii.int().item()) for x_ii in x_i]) for x_i in x]


def str_to_binary_tensor(x):
    return torch.tensor([[int(i) for i in x_i] for x_i in x], dtype=torch.float)


def in_array(element, array):
    for i, e_i in enumerate(array):
        if element.is_equal(e_i):
            return True, i
    return False, None


def to_str_state(obj, rel, mask=None):
    if mask is None:
        mask = torch.ones(obj.shape[0])
    m = (mask == 1)
    n_obj = m.sum()
    mm = torch.outer(m, m)
    obj_str = "-".join(binary_tensor_to_str(obj.bernoulli()[m]))
    rel_str = "-".join([",".join(binary_tensor_to_str(r_i.bernoulli()[mm].reshape(n_obj, n_obj))) for r_i in rel])
    return obj_str + "_" + rel_str


def to_str_action(action):
    to_idx = int(torch.where(action[:, 0] > 0.5)[0])
    to_dx = int(action[to_idx, 1])
    to_dy = int(action[to_idx, 2])

    from_idx = torch.where(action[:, 0] < -0.5)[0]
    if len(from_idx) == 0:
        from_idx = to_idx
        from_dx = 0
        from_dy = 0
    else:
        from_idx = int(from_idx)
        from_dx = int(action[from_idx, 1])
        from_dy = int(action[from_idx, 2])
    act_str = f"{from_idx},{from_dx},{from_dy},{to_idx},{to_dx},{to_dy}"
    return act_str


def to_tensor_state(str_state):
    obj_str, rel_str = str_state.split("_")
    obj = str_to_binary_tensor(obj_str.split("-"))
    rel = torch.stack([str_to_binary_tensor(r_i.split(",")) for r_i in rel_str.split("-")])
    return obj, rel

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

