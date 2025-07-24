import mujoco
import mujoco.viewer
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

model = mujoco.MjModel.from_xml_path("models/panda.xml")
data = mujoco.MjData(model)

# 6D Impedance parameters
K_d = np.diag([200, 200, 200, 10, 10, 10])  # Position (N/m) and orientation (Nm/rad)
B_d = np.diag([30, 30, 30, 3, 3, 3])        # Damping (Ns/m and Nms/rad)

# Desired base position and orientation (identity quaternion)
x_base = np.array([0.5, 0.0, 0.3])
quat_d = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z = no rotation


panda_joint_names = [f"panda0_joint{i+1}" for i in range(7)]
joint_ids = [model.joint(name).qposadr for name in panda_joint_names]

ee_site_name = "end_effector"
ee_site_id = model.site(ee_site_name).id


mujoco.mj_resetData(model, data)
dt = model.opt.timestep
sim_time = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = sim_time

        x_d = x_base + np.array([0.1 * math.sin(2 * math.pi * t / 10), 0, 0])
        xd_dot = np.array([0.1 * (2 * math.pi / 10) * math.cos(2 * math.pi * t / 10), 0, 0])
        omega_d = np.zeros(3)  # Desired angular velocity (static orientation)

        x = data.site_xpos[ee_site_id].copy()
        rot_mat = data.site_xmat[ee_site_id].reshape(3, 3)
        from scipy.spatial.transform import Rotation as R
        quat = R.from_matrix(rot_mat).as_quat()  # xyzw
        quat = np.roll(quat, 1)  # convert to wxyz

        J_pos = np.zeros((3, model.nv))
        J_rot = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, J_pos, J_rot, ee_site_id)

        qvel = data.qvel.copy()
        x_dot = J_pos @ qvel
        omega = J_rot @ qvel

        rot_d = R.from_quat([quat_d[1], quat_d[2], quat_d[3], quat_d[0]])  
        rot_c = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        rot_err = (rot_d * rot_c.inv()).as_rotvec()  

        pos_err = x_d - x
        vel_err = xd_dot - x_dot
        ang_err = rot_err
        ang_vel_err = omega_d - omega

        error_6d = np.concatenate([pos_err, ang_err])
        derror_6d = np.concatenate([vel_err, ang_vel_err])

        wrench = K_d @ error_6d + B_d @ derror_6d

        # Jacobians into 6xnv matrix
        J_full = np.vstack([J_pos, J_rot])

        tau = J_full.T @ wrench

        # add gravity/Coriolis terms
        mujoco.mj_rnePostConstraint(model, data)
        tau += data.qfrc_bias

        data.ctrl[:7] = tau[:7]

        mujoco.mj_step(model, data)
        viewer.sync()

        sim_time += dt
