import numpy
from tvb.simulator.models.base import Model
from tvb.basic.neotraits.api import NArray, List, Range, Final


class MontbrioPazoRoxin_broken(Model):
    r"""
    2D model describing the Ott-Antonsen reduction of infinite all-to-all
    coupled QIF neurons (Theta-neurons) as in [Montbrio_Pazo_Roxin_2015]_.

    The two state variables :math:`r` and :math:`V` represent the average
    firing rate and the average membrane potential of our QIF neurons.

    The equations of the infinite QIF 2D population model read

    .. math::
            \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
            \dot{V} &= 1/\tau (V^2 - \tau^2 \pi^2 r^2 + \eta + J \tau r + I)
    
    Input from the network enters in the :math:`V` variable as 
    :math:`1/\tau(c_r C_r + c_v C_V)` where C is the incomming coupling. In 
    other words, depending on the parameters :math:`c_r`, :math:`c_v` we couple
    the neural masses via the firing rate and/or the membrane potential.
    
    .. [Montbrio_Pazo_Roxin_2015] Montbrió, E., Pazó, D., & Roxin, A. (2015). Macroscopic description for networks of spiking neurons. *Physical Review X*, 5(2), 021028.
    """

    # Define traited attributes for this model, these represent possible kwargs.

    tau = NArray(
        label=r":math:`\tau`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.001, hi=15.0, step=0.01),
        doc="""Characteristic time""",
    )

    I = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""External Current""",
    )

    Delta = NArray(
        label=r":math:`\Delta`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Mean heterogeneous noise""",
    )

    J = NArray(
        label=":math:`J`",
        default=numpy.array([15.0]),
        domain=Range(lo=-25.0, hi=25.0, step=0.0001),
        doc="""Mean Synaptic weight.""",
    )

    eta = NArray(
        label=r":math:`\eta`",
        default=numpy.array([-5.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the
            firing rate variable to itself""",
    )

    Gamma = NArray(
        label=r":math:`\Gamma`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Half-width of synaptic weight distribution""",
    )

    cr = NArray(
        label=":math:`cr`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1, step=0.1),
        doc="""It is the weight on Coupling through variable r.""",
    )

    cv = NArray(
        label=":math:`cv`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=1, step=0.1),
        doc="""It is the weight on Coupling through variable V.""",
    )

    # Informational attribute, used for phase-plane and initial()
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"r": numpy.array([0.0, 2.0]), "V": numpy.array([-2.0, 1.5])},
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup.""",
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"r": numpy.array([0.0, numpy.inf])},
    )

    # TODO should match cvars below..
    coupling_terms = Final(
        label="Coupling terms",
        # how to unpack coupling array
        default=["Coupling_Term_r", "Coupling_Term_V"],
    )

    state_variable_dfuns = Final(
        label="Drift functions",
        default={
            "r": "1/tau * ( Delta / (pi * tau) + 2 * V * r)",
            "V": "1/tau * ( V*V - pi*pi*tau*tau*r*r + eta + J * tau * r + I + cr * Coupling_Term_r + cv * Coupling_Term_V)",
        },
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("r", "V"),
        default=("r", "V"),
        doc="The quantities of interest for monitoring for the Infinite QIF 2D oscillator.",
    )

    parameter_names = List(of=str, label="List of parameters for this model", default="tau Delta eta J I cr cv".split())

    state_variables = ("r", "V")
    _nvar = 2
    # Cvar is the coupling variable.
    cvar = numpy.array([0, 1], dtype=numpy.int32)
    # Stvar is the variable where stimulus is applied.
    stvar = numpy.array([1], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
            2D model describing the Ott-Antonsen reduction of infinite all-to-all
            coupled QIF neurons (Theta-neurons) as in [Montbrio_Pazo_Roxin_2015]_.

            The two state variables :math:`r` and :math:`V` represent the average
            firing rate and the average membrane potential of our QIF neurons.

            The equations of the infinite QIF 2D population model read

            .. math::
                    \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
                    \dot{V} &= 1/\tau (V^2 - \tau^2 \pi^2 r^2 + \eta + J \tau r + I)
        """

        r, V = state_variables

        # [State_variables, nodes]
        I = self.I
        Delta = self.Delta
        Gamma = self.Gamma
        eta = self.eta
        tau = self.tau
        J = self.J
        cr = self.cr
        cv = self.cv

        Coupling_Term_r = coupling[0, :]  # This zero refers to the first element of cvar (r in this case)
        Coupling_Term_V = coupling[1, :]  # This zero refers to the second element of cvar (V in this case)

        derivative = numpy.empty_like(state_variables)

        derivative[0] = 1 / tau * (Delta / (numpy.pi * tau) + 2 * V * r)
        derivative[1] = 1 / tau * (V**2 - numpy.pi**2 * tau**2 * r**2 + eta + J * tau * r - I + cr * Coupling_Term_r + cv * Coupling_Term_V)

        return derivative
