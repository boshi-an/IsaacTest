import isaacgym

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from utils.reformat import omegaconf_to_dict, print_dict
from utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator

from utils.utils import set_np_formatting, set_seed

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

import yaml

import runner


## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)


@hydra.main(config_path="./cfg", config_name="config")
def launch_hydra(cfg: DictConfig) :

	cfg_dict = omegaconf_to_dict(cfg)
	print('\033[1;33m')
	print_dict(cfg_dict)
	print('\033[3;31m')

	# setup numpy formatting
	set_np_formatting()

	# set seed
	cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

	# create simulator
	create_rlgpu_env = get_rlgames_env_creator(
		omegaconf_to_dict(cfg.task),
		cfg.task_name,
		cfg.sim_device,
		cfg.rl_device,
		cfg.graphics_device_id,
		cfg.headless,
		multi_gpu=cfg.multi_gpu
	)

	env = create_rlgpu_env()

if __name__ == "__main__" :
	launch_hydra()