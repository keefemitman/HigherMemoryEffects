import os
import sxs
import scri
import numpy as np
import spherical_functions as sf

from quaternion.calculus import derivative
from quaternion.calculus import indefinite_integral as integrate
from scri.asymptotic_bondi_data.map_to_superrest_frame import MT_to_WM, WM_to_MT


def load_ABD_object(
    path_to_waveforms, radius=None, suffix="", file_format="RPXM", t_junk_cut=None
):
    """
    Load an AsymptoticBondiData (ABD) object.

    Parameters
    ----------
    path_to_waveforms: string
        Path to the directory containing the strain and Weyl scalar data files.
    radius: string, optional [Default: use the lowest radius available]
        Worldtube radius for the waveform files, e.g., "0100".
    suffix: string, optional [Default: ""]
        Add-on to waveform file names, e.g., "_PN_BNS".
    file_format: string, optional [Default: "RPXM"]
        Format of the waveform files, e.g., "SXS" or "RPXM".
    t_junk_cut: float, optional [Default: None]
        Time at which to cut the ABD object. Default is no cut.

    Returns
    -------
    ABD: AsymptoticBondiData
        ABD object containing the strain and Weyl scalar data.

    """
    if not os.path.exists(path_to_waveforms):
        raise ValueError(f"{path_to_waveforms} does not exist!")

    if radius is None:
        radius = (
            [x for x in os.listdir(path_to_waveforms) if "rhOverM" in x][0]
            .split("_R")[-1]
            .split("_")[0]
        )

    abd = scri.SpEC.file_io.create_abd_from_h5(
        h=f"{path_to_waveforms}rhOverM_BondiCce_R{radius}{suffix}.h5",
        Psi4=f"{path_to_waveforms}rMPsi4_BondiCce_R{radius}{suffix}.h5",
        Psi3=f"{path_to_waveforms}r2Psi3_BondiCce_R{radius}{suffix}.h5",
        Psi2=f"{path_to_waveforms}r3Psi2OverM_BondiCce_R{radius}{suffix}.h5",
        Psi1=f"{path_to_waveforms}r4Psi1OverM2_BondiCce_R{radius}{suffix}.h5",
        Psi0=f"{path_to_waveforms}r5Psi0OverM3_BondiCce_R{radius}{suffix}.h5",
        file_format=file_format,
    )
    peak_time = abd.t[
        np.argmax(MT_to_WM(2.0 * abd.sigma.bar.dot, False, scri.hdot).norm())
    ]
    abd.t = abd.t - peak_time

    if not t_junk_cut is None:
        abd = abd.interpolate(abd.t[np.argmin(abs(abd.t - t_junk_cut)) :])

    return abd


def 𝔇inverse(h_mts):
    """
    Compute the \mathfrak{D}^{-1} operator.

    Parameters
    ----------
    h_mts: ModesTimeSeries
        ModesTimeSeries object containing waveform data to be acted on.

    Returns
    -------
    h: ModesTimeSeries
        ModesTimeSeries object containing waveform data that has been acted on by \mathfrak{D}^{-1}.

    """
    h = h_mts.copy()
    s = h.ndarray

    for ell in range(h.ell_min, h.ell_max + 1):
        if ell < 2:
            𝔇inverse_value = 0
        else:
            𝔇inverse_value = -2.0 / ((ell) * (ell + 1))
        s[..., h.index(ell, -ell) : h.index(ell, ell) + 1] *= 𝔇inverse_value

    return h


def zeroth_moment_of_the_news(abd):
    """
    Compute the zeroth moment of the news.

    Parameters
    ----------
    ABD: AsymptoticBondiData
        AsymptoticBondiData to be used in charge/flux calculation.

    Returns
    -------
    soft_flux: ndarray
        The soft flux of the zeroth moment of the news.
    charge: ndarray
        The charge of the zeroth moment of the news.
    hard_flux: ndarray
        The hard flux of the zeroth moment of the news.

    """
    # compute terms
    soft_flux = integrate(np.array(abd.sigma.bar.dot.eth_GHP.eth_GHP.real), abd.t)

    bondi_mass_aspect = abd.mass_aspect()
    charge = np.array(bondi_mass_aspect) - np.array(bondi_mass_aspect)[0]

    hard_flux = integrate(
        np.array(
            abd.sigma.dot.multiply(abd.sigma.bar.dot, truncator=lambda tup: abd.ell_max)
        ),
        abd.t,
    )

    soft_flux[:, :4] *= 0
    charge[:, :4] *= 0
    hard_flux[:, :4] *= 0

    return soft_flux, charge, hard_flux


def first_moment_of_the_news(abd, u_tilde, WZ=True):
    """
    Compute the first moment of the news.

    Parameters
    ----------
    ABD: AsymptoticBondiData
        AsymptoticBondiData to be used in charge/flux calculation.
    utilde: float
        \tilde u to be used in the charge/flux calculation.
    WZ: bool, optional [Default: True]
        Whether or not to add the Wald-Zoupas correction.

    Returns
    -------
    soft_flux: ndarray
        The soft flux of the first moment of the news.
    charge: ndarray
        The charge of the first moment of the news.
    hard_flux0: ndarray
        The hard flux of the first moment of the news.
    hard_flux1: ndarray
        The hard flux 1 of the first moment of the news.

    """
    psi2_tilde = abd.psi2 + abd.sigma.multiply(
        abd.sigma.bar.dot, truncator=lambda tup: abd.ell_max
    )

    energy_flux_density = abd.sigma.dot.multiply(
        abd.sigma.bar.dot, truncator=lambda tup: abd.ell_max
    )
    news_bar_eth_sigma = abd.sigma.bar.dot.multiply(
        abd.sigma.eth_GHP, truncator=lambda tup: abd.ell_max
    )
    sigma_eth_news_bar = abd.sigma.multiply(
        abd.sigma.bar.dot.eth_GHP, truncator=lambda tup: abd.ell_max
    )

    if WZ:
        sigma_bar_eth_sigma = abd.sigma.bar.multiply(
            abd.sigma.eth_GHP, truncator=lambda tup: abd.ell_max
        )
        sigma_eth_sigma_bar = abd.sigma.multiply(
            abd.sigma.bar.eth_GHP, truncator=lambda tup: abd.ell_max
        )
        sigma_bar_eth_news = abd.sigma.bar.multiply(
            abd.sigma.dot.eth_GHP, truncator=lambda tup: abd.ell_max
        )
        news_eth_sigma_bar = abd.sigma.dot.multiply(
            abd.sigma.bar.eth_GHP, truncator=lambda tup: abd.ell_max
        )

    soft_flux = u_tilde * integrate(
        np.array(abd.sigma.bar.dot.eth_GHP.eth_GHP.eth_GHP), abd.t
    ) - integrate(
        np.array(abd.t[:, None] * np.array(abd.sigma.bar.dot.eth_GHP.eth_GHP.eth_GHP)),
        abd.t,
    )

    psi1_tilde = np.array(abd.psi1) + (u_tilde - abd.t[:, None]) * np.array(
        psi2_tilde.eth_GHP
    )

    if WZ:
        psi1_tilde += np.array(0.5 * (sigma_bar_eth_sigma + 3 * sigma_eth_sigma_bar))

    charge = -(psi1_tilde - psi1_tilde[0])

    hard_flux0 = u_tilde * integrate(
        np.array(energy_flux_density.eth_GHP), abd.t
    ) - integrate(
        np.array(abd.t[:, None] * np.array(energy_flux_density.eth_GHP)), abd.t
    )

    if not WZ:
        hard_flux1 = integrate(
            np.array(-news_bar_eth_sigma - 3 * sigma_eth_news_bar), abd.t
        )
    else:
        hard_flux1 = integrate(
            np.array(
                0.5 * (sigma_bar_eth_news + 3 * news_eth_sigma_bar)
                - 0.5 * (news_bar_eth_sigma + 3 * sigma_eth_news_bar)
            ),
            abd.t,
        )

    soft_flux[:, :4] *= 0
    charge[:, :4] *= 0
    hard_flux0[:, :4] *= 0
    hard_flux1[:, :4] *= 0

    return soft_flux, charge, hard_flux0, hard_flux1


def second_moment_of_the_news(abd, u_tilde, WZ=True):
    """
    Compute the second moment of the news.

    Parameters
    ----------
    ABD: AsymptoticBondiData
        AsymptoticBondiData to be used in charge/flux calculation.
    utilde: float
        \tilde u to be used in the charge/flux calculation.
    WZ: bool, optional [Default: True]
        Whether or not to add the Wald-Zoupas correction.

    Returns
    -------
    soft_flux: ndarray
        The soft flux of the second moment of the news.
    charge: ndarray
        The charge of the second moment of the news.
    hard_flux0: ndarray
        The hard flux of the second moment of the news.
    hard_flux1: ndarray
        The hard flux 1 of the second moment of the news.
    hard_flux21_rad: ndarray
        The hard flux 21 rad of the second moment of the news.
    hard_flux21_nonrad: ndarray
        The hard flux 21 nonrad of the second moment of the news.
    hard_flux20: ndarray
        The hard flux 20 rad of the second moment of the news.

    """
    psi2_tilde = abd.psi2 + abd.sigma.multiply(
        abd.sigma.bar.dot, truncator=lambda tup: abd.ell_max
    )

    sigma_psi2_tilde = abd.sigma.multiply(psi2_tilde, truncator=lambda tup: abd.ell_max)

    energy_flux_density = abd.sigma.dot.multiply(
        abd.sigma.bar.dot, truncator=lambda tup: abd.ell_max
    )

    news_bar_eth_sigma = abd.sigma.bar.dot.multiply(
        abd.sigma.eth_GHP, truncator=lambda tup: abd.ell_max
    )
    sigma_eth_news_bar = abd.sigma.multiply(
        abd.sigma.bar.dot.eth_GHP, truncator=lambda tup: abd.ell_max
    )
    bondi_mass_aspect_sigma = abd.mass_aspect().multiply(
        abd.sigma, truncator=lambda tup: abd.ell_max
    )
    sigma_im_eth_eth_sigma_bar = abd.sigma.multiply(
        abd.sigma.bar.eth_GHP.eth_GHP.imag, truncator=lambda tup: abd.ell_max
    )
    sigma_sigma_news_bar = abd.sigma.multiply(
        abd.sigma, truncator=lambda tup: abd.ell_max
    ).multiply(abd.sigma.bar.dot, truncator=lambda tup: abd.ell_max)

    if WZ:
        sigma_bar_eth_sigma = abd.sigma.bar.multiply(
            abd.sigma.eth_GHP, truncator=lambda tup: abd.ell_max
        )
        sigma_eth_sigma_bar = abd.sigma.multiply(
            abd.sigma.bar.eth_GHP, truncator=lambda tup: abd.ell_max
        )
        sigma_bar_eth_news = abd.sigma.bar.multiply(
            abd.sigma.dot.eth_GHP, truncator=lambda tup: abd.ell_max
        )
        news_eth_sigma_bar = abd.sigma.dot.multiply(
            abd.sigma.bar.eth_GHP, truncator=lambda tup: abd.ell_max
        )

    psi1_tilde = np.array(abd.psi1) + (u_tilde - abd.t[:, None]) * np.array(
        psi2_tilde.eth_GHP
    )
    if WZ:
        psi1_tilde += np.array(0.5 * (sigma_bar_eth_sigma + 3 * sigma_eth_sigma_bar))

    soft_flux = (
        0.5
        * u_tilde**2
        * integrate(np.array(abd.sigma.bar.dot.eth_GHP.eth_GHP.eth_GHP.eth_GHP), abd.t)
        - 0.5
        * u_tilde
        * integrate(
            np.array(
                (2 * abd.t)[:, None]
                * np.array(abd.sigma.bar.dot.eth_GHP.eth_GHP.eth_GHP.eth_GHP)
            ),
            abd.t,
        )
        + 0.5
        * integrate(
            np.array(
                (abd.t**2)[:, None]
                * np.array(abd.sigma.bar.dot.eth_GHP.eth_GHP.eth_GHP.eth_GHP)
            ),
            abd.t,
        )
    )

    psi0_tilde = (
        np.array(abd.psi0)
        + (u_tilde - abd.t[:, None])
        * (np.array(psi1_tilde) + np.array(3 * sigma_psi2_tilde))
        - 0.5 * (u_tilde - abd.t[:, None]) ** 2 * np.array(psi2_tilde.eth_GHP.eth_GHP)
    )

    charge = -(psi0_tilde - psi0_tilde[0])

    hard_flux0 = (
        0.5
        * u_tilde**2
        * integrate(np.array(energy_flux_density.eth_GHP.eth_GHP), abd.t)
        - 0.5
        * u_tilde
        * integrate(
            np.array(
                (2 * abd.t)[:, None] * np.array(energy_flux_density.eth_GHP.eth_GHP)
            ),
            abd.t,
        )
        + 0.5
        * integrate(
            np.array(
                (abd.t**2)[:, None] * np.array(energy_flux_density.eth_GHP.eth_GHP)
            ),
            abd.t,
        )
    )

    if not WZ:
        hard_flux1 = u_tilde * integrate(
            np.array(np.array((-news_bar_eth_sigma - 3 * sigma_eth_news_bar).eth_GHP)),
            abd.t,
        ) - integrate(
            np.array(
                abd.t[:, None]
                * np.array((-news_bar_eth_sigma - 3 * sigma_eth_news_bar).eth_GHP)
            ),
            abd.t,
        )
    else:
        hard_flux1 = u_tilde * integrate(
            np.array(
                (
                    0.5 * (sigma_bar_eth_news + 3 * news_eth_sigma_bar)
                    - 0.5 * (news_bar_eth_sigma + 3 * sigma_eth_news_bar)
                ).eth_GHP
            ),
            abd.t,
        ) - integrate(
            np.array(
                abd.t[:, None]
                * np.array(
                    (
                        0.5 * (sigma_bar_eth_news + 3 * news_eth_sigma_bar)
                        - 0.5 * (news_bar_eth_sigma + 3 * sigma_eth_news_bar)
                    ).eth_GHP
                )
            ),
            abd.t,
        )

    hard_flux21_rad = u_tilde * integrate(
        np.array(np.array(-3 * 1j * sigma_im_eth_eth_sigma_bar.dot)), abd.t
    ) - integrate(
        np.array(abd.t[:, None] * np.array(-3 * 1j * sigma_im_eth_eth_sigma_bar.dot)),
        abd.t,
    )

    hard_flux21_nonrad = u_tilde * integrate(
        np.array(np.array(-3 * bondi_mass_aspect_sigma.dot)), abd.t
    ) - integrate(
        np.array(abd.t[:, None] * np.array(-3 * bondi_mass_aspect_sigma.dot)), abd.t
    )

    hard_flux20 = integrate(np.array(-3 * sigma_sigma_news_bar), abd.t)

    soft_flux[:, :4] *= 0
    charge[:, :4] *= 0
    hard_flux0[:, :4] *= 0
    hard_flux1[:, :4] *= 0
    hard_flux21_rad[:, :4] *= 0
    hard_flux21_nonrad[:, :4] *= 0
    hard_flux20[:, :4] *= 0

    return (
        soft_flux,
        charge,
        hard_flux0,
        hard_flux1,
        hard_flux21_rad,
        hard_flux21_nonrad,
        hard_flux20,
    )
