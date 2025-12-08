# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Robot Inspection Environment
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Navigation-Camera-Direct-v0",
    entry_point=f"{__name__}.navigation_env:Isaac3dnavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env:Isaac3dnavigationEnvCfg",
    },
)
