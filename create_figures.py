# Load the utils file and setup the color palette

from utils import *

import matplotlib.pyplot as plt

plt.style.use("paper.mplstyle")

colors = [
    "#000000",
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#D55E00",
    "#785EF0",
]

# widths for PRL
onecol_w_in = 3.4
twocol_w_in = 7.0625

plt.plot()
plt.clf()
plt.style.use("paper.mplstyle")

# Load the ABD object. This is equivalent to the ExtCCE:0001 simulation in https://data.black-holes.org/waveforms/extcce_catalog.html,
# but has been mapped to the PN BMS frame using the code "map_to_superrest_frame" in the scri package.

abd = load_ABD_object(
    "bbh_q1_nospin/bondi_cce_PN_BMS/", radius="0292", suffix="_PN_BMS", t_junk_cut=-4000
)

abd_LL = load_ABD_object(
    "bbh_q1_nospin/bondi_cce_PN_BMS_LL/", radius="0292", suffix="", t_junk_cut=-4000
)
abd_LL = abd_LL.interpolate(abd.t)

h = MT_to_WM(2.0 * abd.sigma.bar)

h_LL = MT_to_WM(2.0 * abd_LL.sigma.bar)

bianchi_violations = abd.bondi_violation_norms

# Figure 1

fig, axis = plt.subplots(
    1, 2, figsize=(twocol_w_in, twocol_w_in * 0.5), sharex=True, sharey=True
)
plt.subplots_adjust(hspace=0.2, wspace=0.1)

idx1 = np.argmin(abs(abd.t - -500))
idx2 = np.argmin(abs(abd.t - 60)) + 1

labels = [
    r"$\dot{\psi}_{0}/M^{2}$",
    r"$\dot{\psi}_{1}/M^{1}$",
    r"$\dot{\psi}_{2}$",
    r"$\psi_{3}$",
    r"$\psi_{4}M$",
    r"$\mathrm{Im}\left[\psi_{2}\right]/M$",
]
for i in range(len(bianchi_violations)):
    axis[0].plot(
        abd.t[idx1:idx2],
        bianchi_violations[i][idx1:idx2],
        lw=1.2,
        label=labels[i],
        color=colors[i],
    )

axis[1].plot(
    abd.t[idx1:idx2],
    np.sqrt(MT_to_WM(abd.psi0.dot - abd_LL.psi0.dot).norm())[idx1:idx2],
    lw=1.2,
    color=colors[0],
)
axis[1].plot(
    abd.t[idx1:idx2],
    np.sqrt(MT_to_WM(abd.psi1.dot - abd_LL.psi1.dot).norm())[idx1:idx2],
    lw=1.2,
    color=colors[1],
)
axis[1].plot(
    abd.t[idx1:idx2],
    np.sqrt(MT_to_WM(abd.psi2.dot - abd_LL.psi2.dot).norm())[idx1:idx2],
    lw=1.2,
    color=colors[2],
)
axis[1].plot(
    abd.t[idx1:idx2],
    np.sqrt(MT_to_WM(abd.psi3 - abd_LL.psi3).norm())[idx1:idx2],
    lw=1.2,
    color=colors[3],
)
axis[1].plot(
    abd.t[idx1:idx2],
    np.sqrt(MT_to_WM(abd.psi4 - abd_LL.psi4).norm())[idx1:idx2],
    lw=1.2,
    color=colors[4],
)
axis[1].plot(
    abd.t[idx1:idx2],
    np.sqrt(MT_to_WM(abd.psi2.imag - abd_LL.psi2.imag).norm())[idx1:idx2],
    lw=1.2,
    color=colors[5],
)
axis[1].set_yscale("log")
axis[1].set_ylim(bottom=1.01e-9)

axis[0].legend(loc="upper left", ncol=2, frameon=True, fontsize=8)

axis[0].set_xlabel(r"$\left(u - u_{\mathrm{peak}}\right)/M$", fontsize=12)
axis[1].set_xlabel(r"$\left(u - u_{\mathrm{peak}}\right)/M$", fontsize=12)
axis[0].set_ylabel("Absolute Error", fontsize=12)

axis[0].set_title("Bianchi Identity Violations", fontsize=12)
axis[1].set_title("Numerical Error", fontsize=12)

plt.savefig("plots/bianchi_violations.pdf", bbox_inches="tight")


def eth_inverse(A, s):
    """
    Compute the action of \eth^{-1} on a data array.

    """
    A_ethinved = np.zeros_like(A, dtype=complex)
    for L in range(0, int(np.sqrt(A.shape[1]))):
        for M in range(-L, L + 1):
            if (L - s) * (L + s + 1) / 2 <= 0:
                continue
            A_ethinved[:, sf.LM_index(L, M, 0)] = (
                1 / np.sqrt((L - s) * (L + s + 1) / 2) * A[:, sf.LM_index(L, M, 0)]
            )

    return A_ethinved


# Compute the zeroth, first, and second moment of the news terms'
# contribution to the shear

u_tilde = abd.t[:, None]

zero_soft_flux, zero_charge, zero_hard_flux = zeroth_moment_of_the_news(abd)

zero_soft_flux -= zero_soft_flux[-1] - np.array(abd.sigma.bar.eth_GHP.eth_GHP)[-1]
zero_charge -= zero_charge[-1]
zero_hard_flux -= zero_hard_flux[-1] - (zero_soft_flux[-1] - zero_charge[-1])

(
    first_soft_flux,
    first_charge,
    first_hard_flux0,
    first_hard_flux1,
) = first_moment_of_the_news(abd, u_tilde)

first_soft_flux = eth_inverse(derivative(first_soft_flux, abd.t), 0)
first_charge = eth_inverse(derivative(first_charge, abd.t), 0)
first_hard_flux0 = eth_inverse(derivative(first_hard_flux0, abd.t), 0)
first_hard_flux1 = eth_inverse(derivative(first_hard_flux1, abd.t), 0)

(
    second_soft_flux,
    second_charge,
    second_hard_flux0,
    second_hard_flux1,
    second_hard_flux21_rad,
    second_hard_flux21_nonrad,
    second_hard_flux20,
) = second_moment_of_the_news(abd, u_tilde)

second_soft_flux = eth_inverse(
    eth_inverse(derivative(derivative(second_soft_flux, abd.t), abd.t), 0), 1
)
second_charge = eth_inverse(
    eth_inverse(derivative(derivative(second_charge, abd.t), abd.t), 0), 1
)
second_hard_flux0 = eth_inverse(
    eth_inverse(derivative(derivative(second_hard_flux0, abd.t), abd.t), 0), 1
)
second_hard_flux1 = eth_inverse(
    eth_inverse(derivative(derivative(second_hard_flux1, abd.t), abd.t), 0), 1
)
second_hard_flux21_rad = eth_inverse(
    eth_inverse(derivative(derivative(second_hard_flux21_rad, abd.t), abd.t), 0), 1
)
second_hard_flux21_nonrad = eth_inverse(
    eth_inverse(derivative(derivative(second_hard_flux21_nonrad, abd.t), abd.t), 0), 1
)
second_hard_flux20 = eth_inverse(
    eth_inverse(derivative(derivative(second_hard_flux20, abd.t), abd.t), 0), 1
)

terms = [
    zero_soft_flux,
    zero_charge,
    zero_hard_flux,
    first_charge,
    first_hard_flux1,
    second_charge,
    second_hard_flux21_rad,
    second_hard_flux21_nonrad,
    second_hard_flux20,
]

terms_names = [
    r"$\mathrm{Re}\left[\eth^{2}\bar{\sigma}\right]$",
    r"$\Delta m$",
    r"$\int\mathcal{F}_{0}\,du$",
    r"$\frac{\partial}{\partial u}\eth^{-1}\Delta\hat{\psi}_{1}$",
    r"$\eth^{-1}\hat{\mathcal{F}}_{1}$",
    r"$\frac{\partial^{2}}{\partial u^{2}}\eth^{-2}\Delta\hat{\psi}_{0}$",
    r"$\frac{\partial}{\partial u}\eth^{-2}\mathcal{F}_{2,1}^{\mathrm{rad.}}$",
    r"$\frac{\partial}{\partial u}\eth^{-2}\mathcal{F}_{2,1}^{\mathrm{nonrad.}}$",
    r"$\frac{\partial}{\partial u}\eth^{-2}\mathcal{F}_{2,0}$",
]

# Compute the contribution from the M == 0 modes

terms_M0 = []
terms_norms = []
for term in terms:
    term_copy = term.copy()
    for L in range(0, abd.ell_max + 1):
        for M in range(-L, L + 1):
            if L < 2 or M != 0:
                term_copy[:, sf.LM_index(L, M, 0)] *= 0

    terms_M0.append(term_copy)

    terms_norms.append(
        np.sqrt(integrate(np.sum(abs(term_copy) ** 2, axis=1), abd.t)[-1])
    )

terms_M0_sorted = [x for _, x in sorted(zip(terms_norms, terms_M0), reverse=True)]

terms_names_sorted = [x for _, x in sorted(zip(terms_norms, terms_names), reverse=True)]

zero_soft_flux_re = MT_to_WM(abd.sigma.bar.eth_GHP.eth_GHP.real, dataType=scri.psi2)
zero_soft_flux_im = MT_to_WM(abd.sigma.bar.eth_GHP.eth_GHP.imag, dataType=scri.psi2)

zero_soft_flux_re_M0 = zero_soft_flux_re.copy()
zero_soft_flux_im_M0 = zero_soft_flux_im.copy()
for L in range(0, abd.ell_max + 1):
    for M in range(-L, L + 1):
        if L < 2 or M != 0:
            zero_soft_flux_re_M0.data[
                :, sf.LM_index(L, M, zero_soft_flux_re.ell_min)
            ] *= 0
            zero_soft_flux_im_M0.data[
                :, sf.LM_index(L, M, zero_soft_flux_im.ell_min)
            ] *= 0

# Figure 2

fig, axis = plt.subplots(3, 1, figsize=(twocol_w_in, twocol_w_in * 0.5))
plt.subplots_adjust(hspace=0.2)

fig.align_ylabels(axis)

idx1 = np.argmin(abs(abd.t - -200))
idx2 = np.argmin(abs(abd.t - 60)) + 1

l_styles = ["-", "--", "dotted", "-."]

plt.suptitle(r"hierarchy of $m=0$ contributions", y=0.95, fontsize=12)

axis[0].plot(
    abd.t[idx1:idx2],
    np.sqrt(zero_soft_flux_re_M0.norm())[idx1:idx2],
    lw=1.2,
    ls=l_styles[0],
    color=colors[0],
    label=r"$\mathrm{Re}\left[\eth^{2}\bar{\sigma}\right]$",
)
axis[0].plot(
    abd.t[idx1:idx2],
    np.sqrt(np.sum(abs(terms_M0_sorted[0]) ** 2, axis=1))[idx1:idx2],
    lw=1.2,
    ls=l_styles[1],
    color=colors[2],
    label=terms_names_sorted[0],
)

axis[0].set_yscale("log")
axis[0].set_xticklabels([])
axis[0].set_ylabel(r"$||\eth^{2}\bar{\sigma}||_{L^{2}}/M$", fontsize=12)
axis[0].legend(loc="upper left", frameon=True, framealpha=1, fontsize=8)
axis[0].set_ylim(bottom=1e-2, top=2e-1)

for i in range(3):
    axis[1].plot(
        abd.t[idx1:idx2],
        np.sqrt(np.sum(abs(terms_M0_sorted[2 + i]) ** 2, axis=1))[idx1:idx2],
        lw=1.2,
        ls=l_styles[i],
        color=colors[3 + i],
        label=terms_names_sorted[2 + i],
    )
axis[1].plot(
    abd.t[idx1:idx2],
    np.sqrt(zero_soft_flux_im_M0.norm())[idx1:idx2],
    lw=1.2,
    ls=l_styles[3],
    color=colors[1],
    label=r"$\mathrm{Im}\left[\eth^{2}\bar{\sigma}\right]$",
)
axis[1].set_yscale("log")
axis[1].set_xticklabels([])
axis[1].set_ylabel(r"$||\eth^{2}\bar{\sigma}||_{L^{2}}/M$", fontsize=12)
axis[1].legend(loc="upper left", frameon=True, framealpha=1, fontsize=8, ncol=2)

for i in range(4):
    axis[2].plot(
        abd.t[idx1:idx2],
        np.sqrt(np.sum(abs(terms_M0_sorted[5 + i]) ** 2, axis=1))[idx1:idx2],
        lw=1.2,
        ls=l_styles[i],
        color=colors[3 + i],
        label=terms_names_sorted[5 + i],
    )
axis[2].set_yscale("log")
axis[2].set_xlabel(r"$\left(u-u_{\mathrm{peak}}\right)/M$", fontsize=12)
axis[2].set_ylabel(r"$||\eth^{2}\bar{\sigma}||_{L^{2}}/M$", fontsize=12)
axis[2].legend(loc="upper left", frameon=True, framealpha=1, fontsize=8, ncol=2)

plt.savefig("plots/term_M0_hierarchy.pdf", bbox_inches="tight")

# Figure 3

fig, axis = plt.subplots(
    1, 2, figsize=(twocol_w_in, twocol_w_in * 0.4), sharex=True, sharey=True
)
plt.subplots_adjust(wspace=0.1)

idx1 = np.argmin(abs(abd.t - -200))
idx2 = np.argmin(abs(abd.t - 60)) + 1

factors = [1, 1e2, 1e4]
extra_names = ["", r"$10^{2}\,\times\,$", r"$10^{4}\,\times\,$"]
for i, idx in enumerate([2, 4, 8]):
    axis[0].plot(
        abd.t[idx1:idx2],
        factors[i] * terms[idx][idx1:idx2, sf.LM_index(2, 0, 0)].real,
        label=extra_names[i] + terms_names[idx],
        color=colors[i],
        lw=1.2,
    )
axis[0].legend(loc="upper left", frameon=True, fontsize=8)

factors = [20, 20 * 1e2]
extra_names = ["", r"$10^{2}\,\times\,$"]
for i, idx in enumerate([4, 8]):
    axis[1].plot(
        abd.t[idx1:idx2],
        factors[i] * terms[idx][idx1:idx2, sf.LM_index(3, 0, 0)].imag,
        label=extra_names[i] + terms_names[idx],
        color=colors[1 + i],
        lw=1.2,
    )
axis[1].legend(loc="upper left", frameon=True, fontsize=8)

axis[0].set_xlabel(r"$(u-u_{\mathrm{peak}})/M$", fontsize=12)
axis[1].set_xlabel(r"$(u-u_{\mathrm{peak}})/M$", fontsize=12)
axis[0].set_ylabel(r"contribution to $\eth^{2}\bar{\sigma}_{(\ell,m)}/M$", fontsize=12)

axis[0].set_title(r"real part of $(\ell,m)=(2,0)$", fontsize=12, y=1.02)
axis[1].set_title(
    r"imaginary part of $(3,0)\times(2\times10^{1})$", fontsize=12, y=1.02
)

plt.savefig("plots/memory_structure.pdf", bbox_inches="tight")

# Figure 4

fig, axis = plt.subplots(2, 3, figsize=(twocol_w_in, twocol_w_in * 0.6), sharex=True)
plt.subplots_adjust(hspace=0.2, wspace=0.1)

idx1 = np.argmin(abs(abd.t - -200))
idx2 = np.argmin(abs(abd.t - 60)) + 1

for i, idx in enumerate([0, 2, 1]):
    axis[0][0].plot(
        abd.t[idx1:idx2],
        terms[idx][idx1:idx2, sf.LM_index(2, 0, 0)].real,
        label=terms_names[idx],
        color=colors[i],
        lw=1.2,
    )
axis[0][0].legend(loc="upper left", frameon=True, fontsize=8, framealpha=1)

ylim0 = (-0.125, 0.16)
axis[0][0].set_ylim(ylim0)

for i, idx in enumerate([4, 3]):
    axis[0][1].plot(
        abd.t[idx1:idx2],
        6 * terms[idx][idx1:idx2, sf.LM_index(2, 0, 0)].real,
        label=terms_names[idx],
        color=colors[1 + i],
        lw=1.2,
    )
axis[0][1].legend(loc="upper left", frameon=True, fontsize=8, framealpha=1)
axis[0][1].set_ylim(ylim0)

for i, idx in enumerate([7, 5]):
    if idx == 5:
        axis[0][2].plot(
            abd.t[idx1:idx2],
            6 * terms[idx][idx1:idx2, sf.LM_index(2, 0, 0)].real,
            label=terms_names[idx],
            color=colors[1 + i],
            lw=1.2,
        )
    else:
        axis[0][2].plot(
            abd.t[idx1:idx2],
            6 * (terms[6] + terms[7] + terms[8])[idx1:idx2, sf.LM_index(2, 0, 0)].real,
            label="$\mathcal{F}_{2,n}$ flux terms",
            color=colors[1 + i],
            lw=1.2,
        )
axis[0][2].legend(loc="upper left", frameon=True, fontsize=8, framealpha=1)
axis[0][2].set_ylim(ylim0)

for i, idx in enumerate([0, 4, 3]):
    if i == 0:
        axis[1][1].plot(
            abd.t[idx1:idx2],
            np.array(abd.sigma.bar.eth_GHP.eth_GHP)[
                idx1:idx2, sf.LM_index(3, 0, 0)
            ].imag,
            label=r"$\mathrm{Im}\left[\eth^{2}\bar{\sigma}\right]$",
            color=colors[i],
            lw=1.2,
        )
    else:
        axis[1][1].plot(
            abd.t[idx1:idx2],
            terms[idx][idx1:idx2, sf.LM_index(3, 0, 0)].imag,
            label=terms_names[idx],
            color=colors[i],
            lw=1.2,
        )
axis[1][1].legend(loc="upper left", frameon=True, fontsize=8, framealpha=1)

ylim1 = (-0.006, 0.008)
axis[1][1].set_ylim(ylim1)

for i, idx in enumerate([7, 5]):
    if idx == 5:
        axis[1][2].plot(
            abd.t[idx1:idx2],
            3 * terms[idx][idx1:idx2, sf.LM_index(3, 0, 0)].imag,
            label=terms_names[idx],
            color=colors[1 + i],
            lw=1.2,
        )
    else:
        axis[1][2].plot(
            abd.t[idx1:idx2],
            3 * (terms[6] + terms[7] + terms[8])[idx1:idx2, sf.LM_index(3, 0, 0)].imag,
            label="$\mathcal{F}_{2,n}$ flux terms",
            color=colors[1 + i],
            lw=1.2,
        )
axis[1][2].legend(loc="upper left", frameon=True, fontsize=8, framealpha=1)
axis[1][2].set_ylim(ylim1)

axis[1][1].set_xlabel(r"$(u-u_{\mathrm{peak}})/M$", fontsize=12)
axis[1][2].set_xlabel(r"$(u-u_{\mathrm{peak}})/M$", fontsize=12)
axis[0][0].set_ylabel(r"real part of $(2,0)$", fontsize=12)
axis[1][1].set_ylabel(r"imaginary part of $(3,0)$", fontsize=12)

axis[0][1].set_title(r"$\times(6\times10^{0})$", fontsize=12)
axis[0][2].set_title(r"$\times(6\times10^{0})$", fontsize=12)
axis[1][2].set_title(r"$\times(3\times10^{0})$", fontsize=12)

axis[0][1].set_yticklabels([])
axis[0][2].set_yticklabels([])
axis[1][2].set_yticklabels([])
axis[1][0].set_visible(False)

plt.savefig("plots/memory_contributions.pdf", bbox_inches="tight")
