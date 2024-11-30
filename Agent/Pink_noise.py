"""
Adopted from Eberhard, Onno and Hollenstein, Jakob and Pinneri, Cristina and Martius, Georg:
Pink Noise Is All You Need: Colored Noise Exploration in Deep Reinforcement Learning
https://github.com/martius-lab/pink-noise-rl
"""


import numpy as np
import Agent.colorednoise as cn

class ColoredNoiseProcess():
    """Infinite colored noise process.

    Implemented as a buffer: every `size[-1]` samples, a cut to a new time series starts. As this cut influences the
    PSD of the combined signal, the maximum period (1 / low-frequency cutoff) can be specified.

    Methods
    -------
    sample(T=1)
        Sample `T` timesteps from the colored noise process.
    reset()
        Reset the buffer with a new time series.
    """
    def __init__(self, beta, size, scale=1, max_period=None, rng=None):
        """Infinite colored noise process.

        Implemented as a buffer: every `size[-1]` samples, a cut to a new time series starts. As this cut influences
        the PSD of the combined signal, the maximum period (1 / low-frequency cutoff) can be specified.

        Parameters
        ----------
        beta : float
            Exponent of colored noise power-law spectrum.
        size : int or tuple of int
            Shape of the sampled colored noise signals. The last dimension (`size[-1]`) specifies the time range, and
            is thus ths maximum possible correlation length of the combined signal.
        scale : int, optional, by default 1
            Scale parameter with which samples are multiplied
        max_period : float, optional, by default None
            Maximum correlation length of sampled colored noise singals (1 / low-frequency cutoff). If None, it is
            automatically set to `size[-1]` (the sequence length).
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        """
        self.beta = beta
        if max_period is None:
            self.minimum_frequency = 0
        else:
            self.minimum_frequency = 1 / max_period
        self.scale = scale
        self.rng = rng

        # The last component of size is the time index
        try:
            self.size = list(size)
        except TypeError:
            self.size = [size]
        self.time_steps = self.size[-1]

        # Fill buffer and reset index
        self.reset()

    def reset(self):
        """Reset the buffer with a new time series."""
        self.buffer = cn.powerlaw_psd_gaussian(
                exponent=self.beta, size=self.size, fmin=self.minimum_frequency, rng=self.rng)
        self.idx = 0

    def sample(self, T=1):
        """
        Sample `T` timesteps from the colored noise process.

        The buffer is automatically refilled when necessary.

        Parameters
        ----------
        T : int, optional, by default 1
            Number of samples to draw

        Returns
        -------
        array_like
            Sampled vector of shape `(*size[:-1], T)`
        """
        n = 0
        ret = []
        while n < T:
            if self.idx >= self.time_steps:
                self.reset()
            m = min(T - n, self.time_steps - self.idx)
            ret.append(self.buffer[..., self.idx:(self.idx + m)])
            n += m
            self.idx += m

        ret = self.scale * np.concatenate(ret, axis=-1)
        return ret if n > 1 else ret[..., 0]


class ColoredActionNoise:
    def __init__(self, beta, sigma, seq_len, action_dim=None, rng=None):
        """Action noise from a colored noise process.

        Parameters
        ----------
        beta : float or array_like
            Exponent(s) of colored noise power-law spectra. If it is a single float, then `action_dim` has to be
            specified and the noise will be sampled in a vectorized manner for each action dimension. If it is
            array_like, then it specifies one beta for each action dimension. This allows different betas for different
            action dimensions, but sampling might be slower for high-dimensional action spaces.
        sigma : float or array_like
            Noise scale(s) of colored noise signals. Either a single float to be used for all action dimensions, or
            an array_like of the same dimensionality as the action space (one scale for each action dimension).
        seq_len : int
            Length of sampled colored noise signals. If sampled for longer than `seq_len` steps, a new
            colored noise signal of the same length is sampled. Should usually be set to the episode length
            (horizon) of the RL task.
        action_dim : int, optional
            Dimensionality of the action space. If passed, `beta` has to be a single float and the noise will be
            sampled in a vectorized manner for each action dimension.
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        """
        super().__init__()
        assert (action_dim is not None) == np.isscalar(beta), \
            "`action_dim` has to be specified if and only if `beta` is a scalar."

        self.sigma = np.full(action_dim or len(beta), sigma) if np.isscalar(sigma) else np.asarray(sigma)

        if np.isscalar(beta):
            self.beta = beta
            self.gen = ColoredNoiseProcess(beta=self.beta, scale=self.sigma, size=(action_dim, seq_len), rng=rng)
        else:
            self.beta = np.asarray(beta)
            self.gen = [ColoredNoiseProcess(beta=b, scale=s, size=seq_len, rng=rng)
                        for b, s in zip(self.beta, self.sigma)]

    def __call__(self) -> np.ndarray:
        return self.gen.sample() if np.isscalar(self.beta) else np.asarray([g.sample() for g in self.gen])

    def __repr__(self) -> str:
        return f"ColoredActionNoise(beta={self.beta}, sigma={self.sigma})"

class PinkActionNoise(ColoredActionNoise):
    def __init__(self, sigma, seq_len, action_dim, rng=None):
        """Action noise from a pink noise process.

        Parameters
        ----------
        sigma : float or array_like
            Noise scale(s) of colored noise signals. Either a single float to be used for all action dimensions, or
            an array_like of the same dimensionality as the action space (one scale for each action dimension).
        seq_len : int
            Length of sampled pink noise signals. If sampled for longer than `seq_len` steps, a new
            pink noise signal of the same length is sampled. Should usually be set to the episode length
            (horizon) of the RL task.
        action_dim : int
            Dimensionality of the action space.
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        """
        super().__init__(1, sigma, seq_len, action_dim, rng)