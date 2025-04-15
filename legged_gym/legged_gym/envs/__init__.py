from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils.task_registry import task_registry

from .base.g1_ground import LeggedRobot as LeggedRobotGround
from legged_gym.envs.g1.g1_config_ground import G1Cfg as G1CfgGround
from legged_gym.envs.g1.g1_config_ground import G1CfgPPO as G1CfgPPOGround

from .base.g1_platform import LeggedRobot as LeggedRobotPlatform
from legged_gym.envs.g1.g1_config_platform import G1Cfg as G1CfgPlatform
from legged_gym.envs.g1.g1_config_platform import G1CfgPPO as G1CfgPPOPlatform

from .base.g1_wall import LeggedRobot as LeggedRobotWall
from legged_gym.envs.g1.g1_config_wall import G1Cfg as G1WallCfgWall
from legged_gym.envs.g1.g1_config_wall import G1CfgPPO as G1WallCfgPPOWall

from .base.g1_slope import LeggedRobot as LeggedRobotSlope
from legged_gym.envs.g1.g1_config_slope import G1Cfg as G1SlopeCfgSlope
from legged_gym.envs.g1.g1_config_slope import G1CfgPPO as G1SlopeCfgPPOSlope


task_registry.register( "g1_ground", LeggedRobotGround, G1CfgGround(), G1CfgPPOGround())
task_registry.register( "g1_platform", LeggedRobotPlatform, G1CfgPlatform(), G1CfgPPOPlatform())
task_registry.register( "g1_wall", LeggedRobotWall, G1WallCfgWall(), G1WallCfgPPOWall())
task_registry.register( "g1_slope", LeggedRobotSlope, G1SlopeCfgSlope(), G1SlopeCfgPPOSlope())