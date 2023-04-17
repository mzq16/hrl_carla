from importlib import import_module
import numpy as np


def load_entry_point(name):
    mod_name, attr_name = name.split(":")
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

def aug_obs_z(obs, z, number_z):
    z_onehot = np.zeros(number_z)
    z_onehot[z] = 1
    x = np.concatenate([obs, z_onehot])
    return x

def split_aug_z(aug_z, number_z):
    pass
