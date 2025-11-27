import numpy
import jax.numpy as np
from mpr_config import MPRConfig
import vbjax as vb
import tvb.simulator.lab as tvbl


class VBJaxModel:
    # Lovely, can't do euler on delays, can't do coupling (on it's own) without delays

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
        self.delay_helper = vb.make_delay_helper(
            weights=np.log(self.config.conn.weights + 1),
            lengths=self.config.conn.tract_lengths,
            dt=self.config.dt,
        )

    def run(self):

        def network(state, _):
            # In VBJax, connectivity handling is up to the user
            # Here's an implementation mirroring other VBJax models
            coupling = self.config.coupling_strength * self.config.conn.weights @ state.T
            coupling = coupling.T
            return vb.mpr_dfun(state, coupling, self.parameters)

        def noise_func(state):
            noise = tvbl.noise.Additive(
                nsig=numpy.r_[self.config.noise],
                noise_seed=self.config.noise_seed,
            )
            noise.dt = self.config.dt
            noise.configure()
            # Unscale noise, because vbjax scales it again
            unscaled_noise = noise.generate(state.shape) / np.sqrt(self.config.dt)

            # In tvb, this is handled by integrator

            return unscaled_noise, noise.gfun(state)

        squeezed_init_conds = np.squeeze(self.config.init_cond)

        # noise_array, noise_scaler = noise_func(squeezed_init_conds)
        noise_array = vb.randn(*squeezed_init_conds.shape)

        step, _ = vb.make_sde(
            dt=self.config.dt,
            dfun=network,
            gfun=self.config.noise,
            return_euler=True,
        )

        return step(squeezed_init_conds, noise_array, self.parameters)[0]


if __name__ == "__main__":
    config = MPRConfig()
    print(config.tau)
    config.init_config_for_connectivity()
    print(VBJaxModel(config).run())
