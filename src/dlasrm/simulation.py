import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from numba import njit, prange, void
from numba.types import Tuple  # type: ignore
from numpy.random import randint, seed
from numpy.typing import NDArray


@njit(void(nb.int64))
def seed_env(state: int) -> None:
    seed(state)


@njit(nb.int64[:](nb.int64))
def seed_gen(size: int) -> NDArray[np.int64]:
    seeds = np.empty(size, dtype=np.int64)
    seed_env(1)

    i = 0
    while i < size:
        candidate = randint(1, 9_999)
        if candidate not in seeds:
            seeds[i] = candidate
            i += 1

    return seeds


@njit(nb.float64[:](nb.int64, nb.float64))
def calculate_emiss_durs(n: int, lifetime_ns: float) -> NDArray[np.float64]:
    return np.log(np.random.random(n)) * -lifetime_ns


@njit(nb.float64[:](nb.int64, nb.float64, nb.float64))
def calculate_laser_pulse_times(n: int, T_ns: float, eff: float) -> NDArray[np.float64]:
    if eff == 1:
        return np.arange(1, n + 1, dtype=np.float64) * T_ns
    else:
        return (
            np.cumsum(
                np.floor(np.log(np.random.random(n)) / np.log(1 - eff)) + 1,
                dtype=np.float64,
            )
            * T_ns
        )


@njit(nb.float64[:](nb.float64[:], nb.float64[:]))
def calculate_emiss_times(
    emiss_durs: NDArray[np.float64], laser_pulse_times: NDArray[np.float64]
) -> NDArray[np.float64]:
    return laser_pulse_times + emiss_durs


# ~1 dark count per 15_000 events with dark count rate = 200/s, laser pulse period = 50ns, emitter efficiency = 1, lifetime = 3ns
@njit(nb.float64[:](nb.float64[:], nb.float64))
def calculate_dark_count_times(
    emiss_times: NDArray[np.float64], rate_ns: float
) -> NDArray[np.float64]:
    dark_count_timeline = np.random.poisson(rate_ns, int(emiss_times[-1]))
    return np.where(dark_count_timeline == 1)[0].astype(np.float64)


# counts of background noise linear to ratio
@njit(nb.float64[:](nb.float64[:], nb.float64))
def calculate_background_times(
    laser_pulse_times: NDArray[np.float64], ratio: float
) -> NDArray[np.float64]:
    laser_pulse_to_background = np.random.choice(
        laser_pulse_times, int(len(laser_pulse_times) * ratio), False
    )
    background_emiss_durs = np.log(np.random.random(len(laser_pulse_to_background)))
    return laser_pulse_to_background + background_emiss_durs


# ~5 secondary emissions per 100 events with laser pulse period = 50ns, laser pulse width = 1ns, lifetime = 3ns
@njit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64, nb.float64))
def calculate_secondary_emiss_times(
    emiss_durs: NDArray[np.float64],
    laser_pulse_times: NDArray[np.float64],
    pulse_width_ns: float,
    lifetime_ns: float,
) -> NDArray[np.float64]:
    diff = pulse_width_ns - emiss_durs
    time_left = np.where(diff > 0, diff, 0)
    pot_secondary_emiss_durs = np.log(np.random.random(len(emiss_durs))) * -lifetime_ns
    secondary_emiss_durs = np.where(
        pot_secondary_emiss_durs <= time_left, pot_secondary_emiss_durs, 0
    )
    secondary_emiss_times = (
        laser_pulse_times * np.where(secondary_emiss_durs != 0, 1, 0)
        + secondary_emiss_durs
    )

    return secondary_emiss_times[secondary_emiss_times != 0]


@njit(nb.float64[:](nb.float64[:], nb.float64[:]))
def incorp_noise(
    noiseless: NDArray[np.float64], noise: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.sort(np.concat((noiseless, noise)))


@njit(Tuple((nb.float64[:], nb.float64[:]))(nb.float64[:], nb.float64[:]))
def get_arrival_times_at_detectors(
    t_1: NDArray[np.float64], t_2: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    t = np.empty(len(t_1) + len(t_2), dtype=np.float64)
    i = 0
    j = 0

    for k in range(len(t)):
        if i == len(t_1):
            t[k:] = t_2[j:]
            break

        if j == len(t_2):
            t[k:] = t_1[i:]
            break

        if t_1[i] <= t_2[j]:
            t[k] = t_1[i]
            i += 1
        else:
            t[k] = t_2[j]
            j += 1

    mask = np.random.random(len(t)) <= 0.5
    return t[mask], t[~mask]


@njit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64))
def calculate_taus(
    t_1: NDArray[np.float64], t_2: NDArray[np.float64], half_int_ns: float
) -> NDArray[np.float64]:
    starts = np.empty(len(t_1), dtype=np.int64)
    ends = np.empty(len(t_1), dtype=np.int64)

    size = 0
    ptr_1 = 0
    ptr_2 = 0
    for i in range(len(t_1)):
        while ptr_1 < len(t_2) and t_2[ptr_1] < t_1[i] - half_int_ns:
            ptr_1 += 1

        ptr_2 = max(ptr_2, ptr_1)

        while ptr_2 < len(t_2) and t_2[ptr_2] < t_1[i] + half_int_ns:
            ptr_2 += 1

        starts[i] = ptr_1
        ends[i] = ptr_2
        size += ptr_2 - ptr_1

    taus = np.empty(size, dtype=np.float64)
    i = 0
    for j in range(len(t_1)):
        for k in range(starts[j], ends[j]):
            taus[i] = t_1[j] - t_2[k]
            i += 1

    return taus


@njit(Tuple((nb.int64[:], nb.float64))(nb.float64[:], nb.float64, nb.int64))
def get_histogram_and_bpp(
    taus: NDArray[np.float64], T_ns: float, bins: int
) -> tuple[NDArray[np.int64], float]:
    hist, edges = np.histogram(taus, bins=bins)
    return hist, np.floor(T_ns / (edges[1] - edges[0]))


@njit(nb.float64(nb.int64[:], nb.float64))
def calculate_g20(hist: NDArray[np.int64], bpp: float) -> float:
    bins = len(hist)
    tau_zero_i = bins // 2
    peak_i = np.arange(bpp, bins, bpp, dtype=np.int64)
    peak_i = peak_i[peak_i != tau_zero_i]

    hbpp = bpp // 2
    length = len(peak_i)
    areas = np.empty(length, dtype=np.float64)
    for i in range(length):
        areas[i] = hist[peak_i[i] - hbpp : peak_i[i] + hbpp].sum()

    return hist[tau_zero_i - hbpp : tau_zero_i + hbpp].sum() / areas.mean()


@njit(
    nb.float64[:](
        nb.int64,
        nb.float64,
        nb.float64,
        nb.float64[:],
        nb.float64[:],
        nb.float64,
        nb.int64,
        nb.int64,
    ),
    parallel=True,
)
def label_gen(
    n: int,
    T_ns: float,
    lifetime_ns: float,
    eff_1s: NDArray[np.float64],
    eff_2s: NDArray[np.float64],
    half_int_ns: float,
    bins: int,
    seed: int,
) -> NDArray[np.float64]:
    length = len(eff_1s)
    g2_zeros = np.empty(length, dtype=np.float64)
    seed_env(seed)

    for i in prange(length):  # type: ignore
        emiss_durs_1 = calculate_emiss_durs(n, lifetime_ns)
        emiss_durs_2 = calculate_emiss_durs(n, lifetime_ns)
        laser_pulse_times_1 = calculate_laser_pulse_times(n, T_ns, eff_1s[i])
        laser_pulse_times_2 = calculate_laser_pulse_times(n, T_ns, eff_2s[i])
        t_1 = calculate_emiss_times(emiss_durs_1, laser_pulse_times_1)
        t_2 = calculate_emiss_times(emiss_durs_2, laser_pulse_times_2)
        t_1, t_2 = get_arrival_times_at_detectors(t_1, t_2)
        taus = calculate_taus(t_1, t_2, half_int_ns)
        hist, bpp = get_histogram_and_bpp(taus, T_ns, bins)
        g2_zeros[i] = calculate_g20(hist, bpp)

    return g2_zeros


@njit(
    nb.int64[:, :](
        nb.int64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64[:],
        nb.float64[:],
        nb.float64,
        nb.int64,
        nb.int64,
    ),
    parallel=True,
)
def input_gen(
    n: int,
    dark_count_rate_ns: float,
    background_ratio: float,
    pulse_width_ns: float,
    T_ns: float,
    lifetime_ns: float,
    eff_1s: NDArray[np.float64],
    eff_2s: NDArray[np.float64],
    half_int_ns: float,
    bins: int,
    seed: int,
) -> NDArray[np.int64]:
    length = len(eff_1s)
    histograms = np.empty((length, bins), dtype=np.int64)
    seed_env(seed)

    for i in prange(length):  # type: ignore
        emiss_durs_1 = calculate_emiss_durs(n, lifetime_ns)
        emiss_durs_2 = calculate_emiss_durs(n, lifetime_ns)
        laser_pulse_times_1 = calculate_laser_pulse_times(n, T_ns, eff_1s[i])
        laser_pulse_times_2 = calculate_laser_pulse_times(n, T_ns, eff_2s[i])
        t_1 = calculate_emiss_times(emiss_durs_1, laser_pulse_times_1)
        t_2 = calculate_emiss_times(emiss_durs_2, laser_pulse_times_2)
        dark_count_times_1 = calculate_dark_count_times(t_1, dark_count_rate_ns)
        dark_count_times_2 = calculate_dark_count_times(t_2, dark_count_rate_ns)
        background_times_1 = calculate_background_times(
            laser_pulse_times_1, background_ratio
        )
        background_times_2 = calculate_background_times(
            laser_pulse_times_2, background_ratio
        )
        secondary_emiss_times_1 = calculate_secondary_emiss_times(
            emiss_durs_1, laser_pulse_times_1, pulse_width_ns, lifetime_ns
        )
        secondary_emiss_times_2 = calculate_secondary_emiss_times(
            emiss_durs_2, laser_pulse_times_2, pulse_width_ns, lifetime_ns
        )
        t_1 = incorp_noise(t_1, dark_count_times_1)
        t_2 = incorp_noise(t_2, dark_count_times_2)
        t_1 = incorp_noise(t_1, background_times_1)
        t_2 = incorp_noise(t_2, background_times_2)
        t_1 = incorp_noise(t_1, secondary_emiss_times_1)
        t_2 = incorp_noise(t_2, secondary_emiss_times_2)
        t_1, t_2 = get_arrival_times_at_detectors(t_1, t_2)
        taus = calculate_taus(t_1, t_2, half_int_ns)
        hist, _ = get_histogram_and_bpp(taus, T_ns, bins)
        histograms[i,] = hist

    return histograms
