import sys

sys.path.append('/mnt/ubuntu/Lab_Files/04_MZQ/carla/PythonAPI/carla/dist/carla-0.9.12-py3.9-linux-x86_64.egg')

import carla


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================

# ==============================================================================
# -- CameraSensor -----------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        attachment = carla.AttachmentType
