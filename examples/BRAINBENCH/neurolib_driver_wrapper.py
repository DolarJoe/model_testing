from neurolib_driver import NeurolibDriver
from suphopf_config import SupHopfConfig
from utils.constants import MODEL_HOPF

# @patrik ked budes toto rozsirovat, myslim si ze v pohode mozes pouzit svoje drivers.


class NeurolibDriverWrapper:
    def __init__(self, config: SupHopfConfig):
        self.config = config
        self._configure_sim()

    def _configure_sim(self):
        """
        Function used to handle configuration of a given simulator
        by using parameters of the Config class
        """
        # @patrik myslim ze sem by viac menej patrilo nieco taketo?
        driver = NeurolibDriver()
        driver.load_model(
            model_name=MODEL_HOPF,
            params={
                "a": self.config.a,
                "w": self.config.w,
            },
        )  # idk ako to pasuje do tvojho navrhu, ci takto ci inak
        driver.setup_network(self.config.conn.weights, self.config.conn.tract_lengths)  # ignoruj warningy, TVB je divne
        # TODO self.config.init_cond pridat

        self.sim = driver

    def run(self):
        """
        The interaction point used for model running.
        We recommend that the unification of simulator outputs is handled here.
        """
        # @patrik asi takto? duration je jeden dt naschval, kedze nas zaujima len jeden krok simulacie pre verifikaciu vysledkov
        self.sim.run_simulation(batch_run=False, duration=self.config.dt, dt=self.config.dt)
        return self.sim.get_results()["x"].reshape((1, -1, 1, 1))
