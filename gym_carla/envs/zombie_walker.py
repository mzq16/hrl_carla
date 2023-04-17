import numpy as np


class ZombieWalker(object):
    def __init__(self, walker, controller, world):
        if type(walker) is int:
            self._walker = world.get_actor(walker)
            self._controller = world.get_actor(controller)
        else:
            self._walker = walker
            self._controller = controller

        self._controller.start()
        self._controller.go_to_location(world.get_random_location_from_navigation())
        self._controller.set_max_speed(1 + np.random.random())


    def clean(self):
        self._controller.stop()
        self._controller.destroy()
        self._walker.destroy()
