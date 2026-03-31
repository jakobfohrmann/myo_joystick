import mujoco
import mujoco.viewer
import numpy as np

# Load model
xml_path = "xml/controller_with_hand.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Print actuator names
print("Actuators:", [model.actuator(i).name for i in range(model.nu)])

# muscles activations set before modification: 
print("Max ctrl before modification:", np.max(data.ctrl))  


# Disable gravity completely
model.opt.gravity[:] = [0, 0, 0]

# Rotating whole hand 
ECU  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ECU") 
FCU  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FCU") 
ECRL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ECRL")
ECRB = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ECRB")
FCR  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FCR")
PQ   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "PQ")
PT   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "PT")

# --- Finger flexors ---
FDP5 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FDP5")
FDS5 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FDS5")

FDP4 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FDP4")
FDS4 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FDS4")

FDP3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FDP3")
FDS3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FDS3")

FDP2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FDP2")
FDS2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FDS2")

# Daumen- und Wrist-Joints identifizieren für Startpose
cmc_abd = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cmc_abduction")
cmc_flex = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cmc_flexion")
mp_flex  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mp_flexion")
ip_flex  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ip_flexion")
pro_sup  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pro_sup")
flexion  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "flexion")
deviation = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "deviation")

# Quick IK: optimiere relevante Thumb-/Wrist-Joints, um THtip auf thumbstick_marker zu bringen
thumb_tip_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "THtip")
stick_site_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "thumbstick_marker")

# Check ob Sites gefunden wurden
if thumb_tip_id < 0:
    raise RuntimeError("THtip site not found in model. Check if hand XML is loaded correctly.")
if stick_site_id < 0:
    raise RuntimeError("thumbstick_marker site not found in model.")
thumb_joints = [cmc_abd, cmc_flex, mp_flex, ip_flex, pro_sup, flexion, deviation]

def set_thumb(q_vars):
    for j_id, q_val in zip(thumb_joints, q_vars):
        data.qpos[j_id] = q_val
    mujoco.mj_forward(model, data)

def dist(q_vars):
    set_thumb(q_vars)
    thumb_pos = data.site_xpos[thumb_tip_id]
    stick_pos = data.site_xpos[stick_site_id]
    return np.linalg.norm(thumb_pos - stick_pos)

# Start aus aktueller Pose
q_vars = np.array([data.qpos[j] for j in thumb_joints])
best = q_vars.copy()
best_d = dist(best)
rng = np.random.default_rng(0)
step = 0.01  # Schrittweite (rad)

for _ in range(400):
    proposal = best + step * rng.normal(size=len(best))
    d = dist(proposal)
    if d < best_d:
        best, best_d = proposal, d

# Finale Pose setzen und berichten
set_thumb(best)
print("Optimized distance:", best_d)
print("Optimized qpos:", best)
print("Corresponding joints:", [model.joint(j).name for j in thumb_joints])

# Reset all muscle activations to zero each frame (initial set)
data.ctrl[:] = 0.0

# Optionale minimale Stabilisierung (anpassbar)
data.ctrl[ECRL] = 0.05
data.ctrl[ECRB] = 0.05
data.ctrl[ECU]  = 0.05
data.ctrl[FCR]  = 0.00
data.ctrl[FCU]  = 0.01
data.ctrl[PQ] = 0.05
data.ctrl[PT] = 0.10

# Finger flexors sehr leicht (kann auch auf 0 gesetzt werden)
data.ctrl[FDP5] = 0.02
data.ctrl[FDS5] = 0.0
data.ctrl[FDP4] = 0.02
data.ctrl[FDS4] = 0.0
data.ctrl[FDP3] = 0.02
data.ctrl[FDS3] = 0.0
data.ctrl[FDP2] = 0.02
data.ctrl[FDS2] = 0.0

# Explizite Thumb-Aktivierungen (aus deiner GUI-Einstellung)
data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "EPL")] = 0.745
data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "EPB")] = 0.76
data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FPL")] = 1.0


mujoco.mj_forward(model, data)

# measure offset between thumbstick and distal thumb
thumbstick_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "thumbstick_base")
thumbstick_pos = data.xpos[thumbstick_body_id]
thumbstick_quat = data.xquat[thumbstick_body_id]
print(f"Thumbstick base position: {thumbstick_pos}")
print(f"Thumbstick base quaternion: {thumbstick_quat}")
distal_thumb_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "distal_thumb")
distal_thumb_pos = data.xpos[distal_thumb_body_id]
distal_thumb_quat = data.xquat[distal_thumb_body_id]
print(f"Distal thumb position: {distal_thumb_pos}")
print(f"Distal thumb quaternion: {distal_thumb_quat}")

position_offset = distal_thumb_pos - thumbstick_pos
print(f"Position offset: {position_offset}")
print(f"Distance: {np.linalg.norm(position_offset):.6f} meters")


with mujoco.viewer.launch(model, data) as viewer:

    while viewer.is_running():

        # Reset all muscle activations to zero each frame
        data.ctrl[:] = 0.0

        # Gleiche minimale Stabilisierung wie initial
        data.ctrl[ECRL] = 0.05
        data.ctrl[ECRB] = 0.05
        data.ctrl[ECU]  = 0.05
        data.ctrl[FCR]  = 0.00
        data.ctrl[FCU] = 0.01
        data.ctrl[PQ] = 0.05
        data.ctrl[PT] = 0.10

        # Finger flexors leicht
        data.ctrl[FDP5] = 0.02
        data.ctrl[FDS5] = 0.0
        data.ctrl[FDP4] = 0.02
        data.ctrl[FDS4] = 0.0
        data.ctrl[FDP3] = 0.02
        data.ctrl[FDS3] = 0.0
        data.ctrl[FDP2] = 0.02
        data.ctrl[FDS2] = 0.0

        # Thumb tonus wie initial
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "EPL")] = 0.745
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "EPB")] = 0.76
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FPL")] = 1.0


        
        mujoco.mj_step(model, data)

    
        viewer.sync()

