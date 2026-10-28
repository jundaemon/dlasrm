import numpy as np
from matplotlib import pyplot as plt

from dlasrm import (
    BACKGROUND_RATIO,
    BINS,
    DARK_COUNT_RATE_NS,
    HALF_INT_NS,
    INPUTS_N,
    LIFETIME_NS,
    PULSE_WIDTH_NS,
    T_NS,
)
from dlasrm.simulation import (
    calculate_background_times,
    calculate_dark_count_times,
    calculate_emiss_durs,
    calculate_emiss_times,
    calculate_laser_pulse_times,
    calculate_secondary_emiss_times,
    calculate_taus,
    get_arrival_times_at_detectors,
    incorp_noise,
    seed_env,
)

if __name__ == "__main__":
    seed_env(1)
    emiss_durs_1 = calculate_emiss_durs(INPUTS_N, LIFETIME_NS)
    laser_pulse_times_1 = calculate_laser_pulse_times(INPUTS_N, T_NS, 1)
    emiss_times_1 = calculate_emiss_times(emiss_durs_1, laser_pulse_times_1)
    dark_count_times_1 = calculate_dark_count_times(emiss_times_1, DARK_COUNT_RATE_NS)
    background_times_1 = calculate_background_times(
        laser_pulse_times_1, BACKGROUND_RATIO
    )
    secondary_emission_times_1 = calculate_secondary_emiss_times(
        emiss_durs_1, laser_pulse_times_1, PULSE_WIDTH_NS, LIFETIME_NS
    )
    noisy_emiss_times_1 = incorp_noise(emiss_times_1, dark_count_times_1)
    noisy_emiss_times_1 = incorp_noise(noisy_emiss_times_1, background_times_1)
    noisy_emiss_times_1 = incorp_noise(noisy_emiss_times_1, secondary_emission_times_1)

    emiss_durs_2 = calculate_emiss_durs(INPUTS_N, LIFETIME_NS)
    laser_pulse_times_2 = calculate_laser_pulse_times(INPUTS_N, T_NS, 1)
    emiss_times_2 = calculate_emiss_times(emiss_durs_2, laser_pulse_times_2)
    dark_count_times_2 = calculate_dark_count_times(emiss_times_2, DARK_COUNT_RATE_NS)
    background_times_2 = calculate_background_times(
        laser_pulse_times_2, BACKGROUND_RATIO
    )
    secondary_emission_times_2 = calculate_secondary_emiss_times(
        emiss_durs_2, laser_pulse_times_2, PULSE_WIDTH_NS, LIFETIME_NS
    )
    noisy_emiss_times_2 = incorp_noise(emiss_times_2, dark_count_times_2)
    noisy_emiss_times_2 = incorp_noise(noisy_emiss_times_2, background_times_2)
    noisy_emiss_times_2 = incorp_noise(noisy_emiss_times_2, secondary_emission_times_2)

    emiss_times_1, emiss_times_2 = get_arrival_times_at_detectors(
        emiss_times_1, emiss_times_2
    )
    noisy_emiss_times_1, noisy_emiss_times_2 = get_arrival_times_at_detectors(
        noisy_emiss_times_1, noisy_emiss_times_2
    )
    taus_1 = calculate_taus(emiss_times_1, emiss_times_2, HALF_INT_NS)
    taus_2 = calculate_taus(noisy_emiss_times_1, noisy_emiss_times_2, HALF_INT_NS)
    hist_1, edges_1 = np.histogram(taus_1, BINS)
    hist_2, edges_2 = np.histogram(taus_2, BINS)

    plt.stairs(hist_1, edges_1, label="noiseless", color="b", fill=True)
    plt.stairs(hist_2, edges_2, label="noise", color="r", fill=False)
    plt.title("counts - taus")
    plt.xlabel("taus")
    plt.ylabel("counts")
    plt.legend()
    plt.savefig("results/noise_viz.png")
