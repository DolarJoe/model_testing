import jax.numpy as np
import vbjax as vb
from mpr_config import MPRConfig


class VBJaxModel:
    def __init__(self, config: MPRConfig):
        self.config = config
        self.parameters = vb.MPRTheta(
            tau=config.tau,
            I=config.I,
            Delta=config.Delta,
            J=config.J,
            eta=config.eta,
            cr=config.cr,
            cv=config.cv,
        )

    def run(self):

        def network(state, _):
            # In VBJax, connectivity handling is up to the user
            # Here's an implementation mirroring other VBJax models
            coupling = self.config.coupling_strength * self.config.conn.weights @ state.T
            coupling = coupling.T
            return vb.mpr_dfun(state, coupling, self.parameters)

        squeezed_init_conds = np.squeeze(self.config.init_cond)

        noise_array = vb.randn(*squeezed_init_conds.shape)

        step, _ = vb.make_sde(
            dt=self.config.dt,
            dfun=network,
            gfun=self.config.noise,
            return_euler=True,
        )

        return step(squeezed_init_conds, noise_array, self.parameters)[0]
