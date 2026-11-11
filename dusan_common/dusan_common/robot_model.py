"""
Canonical robot description constants for the Spot-like quadruped "spot".

Derived from the rig of SpotHPRIG_Rigging_02.fbx (Codyssey asset package):
bone positions were converted to the robot frame (x forward, y left, z up,
origin = centre between the four hip_x joints).  The URDF
(simulation/urdf/spot.urdf.xacro) and every Python module MUST use these
numbers so that kinematics, MPC, IK and the Gazebo model agree.

Leg order everywhere: FL, FR, RL, RR  (index 0..3)
Joint order per leg:  hip_x (abduction, axis +X), hip_y (flexion, axis +Y),
                      knee (axis +Y)
Joint sign convention (zero = leg straight down):
  hip_y > 0 swings the foot backwards, knee < 0 bends the knee backwards
  (Spot style "knees point backwards" for all four legs).
"""
import math

ROBOT_NAME = "spot"
WORLD_NAME = "plant"

LEG_NAMES = ("fl", "fr", "rl", "rr")
JOINT_SUFFIXES = ("hip_x", "hip_y", "knee")
JOINT_NAMES = tuple(f"{leg}_{j}" for leg in LEG_NAMES for j in JOINT_SUFFIXES)  # 12 joints

# ---------------------------------------------------------------------------
# Geometry (metres)
# ---------------------------------------------------------------------------
HIP_X_OFFSET = 0.2245      # |x| of hip_x joints from base origin (front +, rear -)
HIP_Y_OFFSET = 0.044       # |y| of hip_x joints from base origin
HIP_TO_THIGH_Y = 0.090     # lateral offset hip_x joint -> hip_y joint (leg plane |y| = 0.134)
LEG_PLANE_Y = HIP_Y_OFFSET + HIP_TO_THIGH_Y   # 0.134
UPPER_LEG_LENGTH = 0.2256  # hip_y -> knee
LOWER_LEG_LENGTH = 0.2997  # knee  -> foot centre
FOOT_RADIUS = 0.025

# sign of x / y per leg
LEG_SIGN_X = (1.0, 1.0, -1.0, -1.0)
LEG_SIGN_Y = (1.0, -1.0, 1.0, -1.0)


def hip_x_position(leg_index):
    """Position of the hip_x joint of a leg in the base frame."""
    return (LEG_SIGN_X[leg_index] * HIP_X_OFFSET, LEG_SIGN_Y[leg_index] * HIP_Y_OFFSET, 0.0)


def hip_y_position_zero(leg_index):
    """Position of the hip_y joint (hip_x = 0) in the base frame."""
    return (LEG_SIGN_X[leg_index] * HIP_X_OFFSET, LEG_SIGN_Y[leg_index] * LEG_PLANE_Y, 0.0)


# ---------------------------------------------------------------------------
# Joint limits (rad, Nm, rad/s)
# ---------------------------------------------------------------------------
JOINT_LIMITS = {
    "hip_x": {"lower": -0.70, "upper": 0.70, "effort": 45.0, "velocity": 15.0},
    "hip_y": {"lower": -1.60, "upper": 2.60, "effort": 45.0, "velocity": 15.0},
    "knee":  {"lower": -2.80, "upper": -0.20, "effort": 60.0, "velocity": 15.0},
}

# ---------------------------------------------------------------------------
# Mass properties (kg, kg m^2) - used by URDF and MPC
# ---------------------------------------------------------------------------
BASE_MASS = 8.0
HIP_MASS = 0.6
UPPER_LEG_MASS = 0.9
LOWER_LEG_MASS = 0.25
TOTAL_MASS = BASE_MASS + 4 * (HIP_MASS + UPPER_LEG_MASS + LOWER_LEG_MASS)   # 15.0
BASE_BOX = (0.60, 0.22, 0.14)            # collision / inertia box of the trunk
BASE_INERTIA = (0.0453, 0.2531, 0.2723)  # Ixx Iyy Izz of the trunk box
# effective centroidal inertia used by the MPC (trunk + legs in nominal stance)
MPC_BODY_INERTIA = (0.07, 0.32, 0.34)

# ---------------------------------------------------------------------------
# Nominal postures
# ---------------------------------------------------------------------------
NOMINAL_BODY_HEIGHT = 0.34      # base origin above ground while standing
MIN_BODY_HEIGHT = 0.20
MAX_BODY_HEIGHT = 0.44
SIT_BODY_HEIGHT = 0.16
NOMINAL_FOOT_X = HIP_X_OFFSET   # feet below the hips
NOMINAL_FOOT_Y = LEG_PLANE_Y


def nominal_foot_positions(body_height=NOMINAL_BODY_HEIGHT):
    """Nominal foot centre positions in the base frame (list of 4 (x,y,z))."""
    z = -(body_height - FOOT_RADIUS)
    return [(LEG_SIGN_X[i] * NOMINAL_FOOT_X, LEG_SIGN_Y[i] * NOMINAL_FOOT_Y, z) for i in range(4)]


# ---------------------------------------------------------------------------
# Sensors (pose of the sensor link in base_link: x y z roll pitch yaw)
#   Camera links follow the Gazebo convention: the camera looks along +X of
#   its link.  Positive pitch tilts the view DOWN.
# ---------------------------------------------------------------------------
SENSORS = {
    "imu": {"link": "imu_link", "pose": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), "rate": 200.0},
    "depth_camera": {
        "link": "depth_camera_link", "pose": (0.34, 0.0, 0.0, 0.0, 0.60, 0.0),
        "width": 212, "height": 120, "hfov": 1.50, "near": 0.10, "far": 4.0, "rate": 10.0,
        "topic": "/depth_camera/image", "points_topic": "/depth_camera/points",
        "info_topic": "/depth_camera/camera_info",
    },
    "inspection_camera": {
        "link": "inspection_camera_link", "pose": (0.30, 0.0, 0.20, 0.0, -0.30, 0.0),
        "width": 800, "height": 600, "hfov": 0.90, "near": 0.05, "far": 50.0, "rate": 10.0,
        "topic": "/inspection_camera/image", "info_topic": "/inspection_camera/camera_info",
    },
    "thermal_camera": {
        "link": "thermal_camera_link", "pose": (0.30, -0.05, 0.20, 0.0, -0.30, 0.0),
        "width": 320, "height": 240, "hfov": 0.80, "near": 0.10, "far": 50.0, "rate": 5.0,
        "topic": "/thermal_camera/image",
        # L16 image, Kelvin = pixel * THERMAL_RESOLUTION
        "min_temp": 253.15, "max_temp": 655.0,
    },
}
THERMAL_RESOLUTION = 0.01   # Kelvin per L16 count

# ---------------------------------------------------------------------------
# Gazebo topic helpers
# ---------------------------------------------------------------------------


def joint_command_topic(joint_name):
    """Topic (identical on the ROS and Gazebo side) for a joint position command."""
    return f"/{ROBOT_NAME}/joint/{joint_name}/cmd_pos"


GZ_ODOM_TOPIC = f"/model/{ROBOT_NAME}/odometry"
GZ_JOINT_STATE_TOPIC = f"/world/{WORLD_NAME}/model/{ROBOT_NAME}/joint_state"

# Joint position controller (Gazebo JointPositionController in PID force mode,
# <cmd_max>/<cmd_min> = +-joint effort; DESIGN_SPEC §3, tools/physics_check.py)
JOINT_SERVO_P_GAIN = 450.0        # N m/rad
JOINT_SERVO_D_GAIN = 3.0          # N m s/rad (d dt / I_knee << 1 at the 1 ms gz step)
JOINT_SERVO_MAX_VELOCITY = 12.0   # rad/s, controller joint-target rate limit (max_joint_speed)

DEG = math.pi / 180.0
