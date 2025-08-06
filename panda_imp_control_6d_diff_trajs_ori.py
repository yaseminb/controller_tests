import mujoco
import mujoco.viewer
import numpy as np
import math
from scipy.spatial.transform import Rotation as R


model = mujoco.MjModel.from_xml_path("models/panda.xml")
data = mujoco.MjData(model)

K_pos = np.diag([200.0, 200.0, 200.0])
B_pos = np.diag([30.0, 30.0, 30.0])
K_ori = np.diag([10.0, 10.0, 10.0])
B_ori = np.diag([1.0, 1.0, 1.0])

x_base = np.array([0.5, 0.0, 0.3])

panda_joint_names = [f"panda0_joint{i+1}" for i in range(7)]
joint_ids = [model.joint(name).qposadr for name in panda_joint_names]

ee_site_name = "end_effector"
ee_site_id = model.site(ee_site_name).id

mujoco.mj_resetData(model, data)
dt = model.opt.timestep
sim_time = 0.0

def circular_trajectory(t):
    radius = 0.1
    omega = 2 * np.pi / 10
    pos = x_base + np.array([
        radius * np.cos(omega * t),
        radius * np.sin(omega * t),
        0.0
    ])
    vel = np.array([
        -radius * omega * np.sin(omega * t),
        radius * omega * np.cos(omega * t),
        0.0
    ])
    quat_d = np.array([1.0, 0.0, 0.0, 0.0])
    return pos, vel, quat_d

def lissajous_trajectory(t):
    pos = x_base + np.array([
        0.1 * np.sin(2 * np.pi * t / 10),
        0.1 * np.sin(4 * np.pi * t / 10),
        0.05 * np.sin(2 * np.pi * t / 5)
    ])
    vel = np.array([
        0.1 * 2 * np.pi / 10 * np.cos(2 * np.pi * t / 10),
        0.1 * 4 * np.pi / 10 * np.cos(4 * np.pi * t / 10),
        0.05 * 2 * np.pi / 5 * np.cos(2 * np.pi * t / 5)
    ])
    quat_d = np.array([1.0, 0.0, 0.0, 0.0])
    return pos, vel, quat_d


def spiral_trajectory(t):
    r = 0.05 + 0.01 * t
    theta = 2 * np.pi * t / 5
    pos = x_base + np.array([
        r * np.cos(theta),
        r * np.sin(theta),
        0.02 * t
    ])
    vel = np.array([
        -r * np.sin(theta) * (2 * np.pi / 5),
        r * np.cos(theta) * (2 * np.pi / 5),
        0.02
    ])
    quat_d = np.array([1.0, 0.0, 0.0, 0.0])
    return pos, vel, quat_d

def sinusoidal_x_trajectory(t):
    pos = x_base + np.array([0.1 * np.sin(2 * np.pi * t / 10), 0, 0])
    vel = np.array([0.1 * (2 * np.pi / 10) * np.cos(2 * np.pi * t / 10), 0, 0])
    quat_d = np.array([1.0, 0.0, 0.0, 0.0])
    return pos, vel, quat_d

trajectories = {
    "1": ("Sinusoidal X", sinusoidal_x_trajectory),
    "2": ("Circular XY", circular_trajectory),
    "3": ("Lissajous", lissajous_trajectory),
    "4": ("Spiral", spiral_trajectory)
}

print("\n[Select Trajectory Type] Choose 1-4:")
for k, (name, _) in trajectories.items():
    print(f"  {k}: {name}")

selected_key = input("Enter choice (1-4): ").strip()
if selected_key not in trajectories:
    print("Invalid choice, defaulting to Sinusoidal X")
    selected_key = "1"

trajectory_fn = trajectories[selected_key][1]
print(f"Using trajectory: {trajectories[selected_key][0]}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = sim_time

        x_d, xd_dot, quat_d = trajectory_fn(t)

        x = data.site_xpos[ee_site_id].copy()
        rot_mat = data.site_xmat[ee_site_id].reshape(3, 3)
        rot_c = R.from_matrix(rot_mat)
        quat = np.roll(rot_c.as_quat(), 1)

        J_pos = np.zeros((3, model.nv))
        J_rot = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, J_pos, J_rot, ee_site_id)

        qvel = data.qvel.copy()
        x_dot = J_pos @ qvel
        w = J_rot @ qvel

        q_err = R.from_quat(np.roll(quat_d, -1)) * R.from_quat(np.roll(quat, -1)).inv()
        rotvec_err = q_err.as_rotvec()

        F = K_pos @ (x_d - x) + B_pos @ (xd_dot - x_dot)
        T = K_ori @ rotvec_err + B_ori @ (-w)

        wrench = np.concatenate([F, T])
        J_full = np.vstack([J_pos, J_rot])
        tau = J_full.T @ wrench

        mujoco.mj_rnePostConstraint(model, data)
        tau += data.qfrc_bias

        data.ctrl[:7] = tau[:7]

        mujoco.mj_step(model, data)
        viewer.sync()
        sim_time += dt
