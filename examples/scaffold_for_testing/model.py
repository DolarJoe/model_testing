from config import Config

# @patrik ked budes toto rozsirovat, myslim si ze v pohode mozes pouzit svoje drivers.


class ExampleSimulatorWrapper:
    def __init__(self, config: Config):
        self.config = config
        self._configure_sim()

    def _configure_sim(self):
        """
        Function used to handle configuration of a given simulator
        by using parameters of the Config class
        """
        # self.sim = ExampleSimulator(self.config)
        pass

    def run(self):
        """
        The interaction point used for model running.
        We recommend that the unification of simulator outputs is handled here.
        """
        return self.sim.run()
