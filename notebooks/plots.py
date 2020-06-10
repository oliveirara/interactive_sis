#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries and set some parameters:

import warnings

warnings.filterwarnings("ignore")

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator

matplotlib.rc("text", usetex=True)
matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

import mpmath

mpmath.mp.dps = 53

# Vectorize mpmath functions:

vec_sqrt_mpmath = np.vectorize(mpmath.sqrt)
vec_acos_mpmath = np.vectorize(mpmath.acos)
vec_re_mpmath = np.vectorize(mpmath.re)

# Image/Lens/Source Properties:


def image_curve(phi, s, phi_s, e_s, R_0, opn):

    a_s, b_s = 1 / np.sqrt(1 - e_s), 1 / np.sqrt(1 + e_s)

    s_1 = s
    s_2 = 0

    alpha_1, alpha_2 = np.cos(phi), np.sin(phi)

    Q_1 = s_1 + alpha_1
    Q_2 = s_2 + alpha_2
    Q_1_bar = Q_1 * np.cos(phi_s) + Q_2 * np.sin(phi_s)
    Q_2_bar = -Q_1 * np.sin(phi_s) + Q_2 * np.cos(phi_s)

    P_1_bar = np.cos(phi - phi_s)
    P_2_bar = np.sin(phi - phi_s)
    S_bar = b_s ** 2 * P_1_bar ** 2 + a_s ** 2 * P_2_bar ** 2

    x = (1 / S_bar) * (b_s ** 2 * Q_1_bar * P_1_bar + a_s ** 2 * Q_2_bar * P_2_bar)

    root_term = S_bar * R_0 ** 2 - (-Q_1_bar * P_2_bar + Q_2_bar * P_1_bar) ** 2
    if root_term >= 0:
        w = (2 * a_s * b_s / S_bar) * np.sqrt(root_term)
    else:
        w = np.nan

    if opn == "x":
        return x
    elif opn == "w":
        return w
    elif opn == "x*w":
        return x * w
    elif opn == "in":
        return x - (1 / 2) * w
    elif opn == "out":
        return x + (1 / 2) * w
    elif opn == "all":
        return [x, w, x * w]
    else:
        return print('You should choose "x", "w", "x*w", "in", "out" or "all".')


vec_image_curve = np.vectorize(image_curve)


def source_curve(phi, s, phi_s, e_s, R_0):
    a_s, b_s = 1 / np.sqrt(1 - e_s), 1 / np.sqrt(1 + e_s)
    return R_0 * np.array(
        [
            (np.cos(phi_s) * np.cos(phi) * a_s)
            - (np.sin(phi_s) * np.sin(phi) * b_s)
            + s / R_0,
            (np.cos(phi_s) * np.sin(phi) * b_s) + (np.cos(phi) * np.sin(phi_s) * a_s),
        ]
    )


def tangential_pseudo_caustic_curve(phi):
    u = -np.cos(phi)
    v = -np.sin(phi)
    return [u, v]


# Angles:


def alpha(s, phi_s, e_s, R_0):
    result = (
        (s ** 4) * (1 - e_s ** 2) ** 2
        + 2 * (R_0 ** 4) * e_s * (e_s + np.cos(2 * phi_s))
        - (R_0 ** 2) * (s ** 2) * (1 - e_s ** 2) * (1 + 3 * e_s * np.cos(2 * phi_s))
    )
    return result


def beta(s, phi_s, e_s, R_0):
    root_term = (
        R_0 ** 6
        * e_s ** 2
        * (-1 + e_s ** 2)
        * (R_0 ** 2 - s ** 2 + (s ** 2) * e_s * np.cos(2 * phi_s))
        * np.sin(2 * phi_s) ** 2
    )
    if root_term >= 0:
        return 2 * np.sqrt(root_term)
    else:
        return 2 * vec_sqrt_mpmath(root_term)


def gamma(s, phi_s, e_s, R_0):
    result = (
        4 * (R_0 ** 4) * (e_s ** 2)
        + (s ** 4) * (-1 + e_s ** 2) ** 2
        + 4 * (R_0 ** 2) * (s ** 2) * e_s * (-1 + e_s ** 2) * np.cos(2 * phi_s)
    )
    return result


def phi_1(s, phi_s, e_s, R_0):

    Alpha = alpha(s, phi_s, e_s, R_0)
    Beta = beta(s, phi_s, e_s, R_0)
    Gamma = gamma(s, phi_s, e_s, R_0)

    Inner_term = (Alpha - Beta) / Gamma

    if Inner_term >= 0:
        return np.real(-np.arccos(-np.sqrt(Inner_term)))
    else:
        return vec_re_mpmath(-vec_acos_mpmath(-vec_sqrt_mpmath(Inner_term)))


def phi_2(s, phi_s, e_s, R_0):

    Alpha = alpha(s, phi_s, e_s, R_0)
    Beta = beta(s, phi_s, e_s, R_0)
    Gamma = gamma(s, phi_s, e_s, R_0)

    Inner_term = (Alpha - Beta) / Gamma

    if Inner_term >= 0:
        return np.real(-np.arccos(np.sqrt(Inner_term)))
    else:
        return vec_re_mpmath(-vec_acos_mpmath(vec_sqrt_mpmath(Inner_term)))


def phi_3(s, phi_s, e_s, R_0):

    Alpha = alpha(s, phi_s, e_s, R_0)
    Beta = beta(s, phi_s, e_s, R_0)
    Gamma = gamma(s, phi_s, e_s, R_0)

    Inner_term = (Alpha + Beta) / Gamma

    if Inner_term >= 0:
        return np.real(-np.arccos(-np.sqrt(Inner_term)))
    else:
        return vec_re_mpmath(-vec_acos_mpmath(-vec_sqrt_mpmath(Inner_term)))


def phi_4(s, phi_s, e_s, R_0):

    Alpha = alpha(s, phi_s, e_s, R_0)
    Beta = beta(s, phi_s, e_s, R_0)
    Gamma = gamma(s, phi_s, e_s, R_0)

    Inner_term = (Alpha + Beta) / Gamma

    if Inner_term >= 0:
        return np.real(-np.arccos(np.sqrt(Inner_term)))
    else:
        return vec_re_mpmath(-vec_acos_mpmath(vec_sqrt_mpmath(Inner_term)))


def phi_5(s, phi_s, e_s, R_0):

    Alpha = alpha(s, phi_s, e_s, R_0)
    Beta = beta(s, phi_s, e_s, R_0)
    Gamma = gamma(s, phi_s, e_s, R_0)

    Inner_term = (Alpha - Beta) / Gamma

    if Inner_term >= 0:
        return np.real(np.arccos(-np.sqrt(Inner_term)))
    else:
        return vec_re_mpmath(vec_acos_mpmath(-vec_sqrt_mpmath(Inner_term)))


def phi_6(s, phi_s, e_s, R_0):

    Alpha = alpha(s, phi_s, e_s, R_0)
    Beta = beta(s, phi_s, e_s, R_0)
    Gamma = gamma(s, phi_s, e_s, R_0)

    Inner_term = (Alpha - Beta) / Gamma

    if Inner_term >= 0:
        return np.real(np.arccos(np.sqrt(Inner_term)))
    else:
        return vec_re_mpmath(vec_acos_mpmath(vec_sqrt_mpmath(Inner_term)))


def phi_7(s, phi_s, e_s, R_0):

    Alpha = alpha(s, phi_s, e_s, R_0)
    Beta = beta(s, phi_s, e_s, R_0)
    Gamma = gamma(s, phi_s, e_s, R_0)

    Inner_term = (Alpha + Beta) / Gamma

    if Inner_term >= 0:
        return np.real(np.arccos(-np.sqrt(Inner_term)))
    else:
        return vec_re_mpmath(vec_acos_mpmath(-vec_sqrt_mpmath(Inner_term)))


def phi_8(s, phi_s, e_s, R_0):

    Alpha = alpha(s, phi_s, e_s, R_0)
    Beta = beta(s, phi_s, e_s, R_0)
    Gamma = gamma(s, phi_s, e_s, R_0)

    Inner_term = (Alpha + Beta) / Gamma

    if Inner_term >= 0:
        return np.real(np.arccos(np.sqrt(Inner_term)))
    else:
        return vec_re_mpmath(vec_acos_mpmath(vec_sqrt_mpmath(Inner_term)))


# Plot Functions:
#################


def plot(s, phi_s, e_s, R_0):

    phi = np.arange(0, 2 * np.pi, 0.001)

    # Lens Place:
    #############

    # Tangential Critical Curves
    tcx = 1 * np.cos(phi)
    tcy = 1 * np.sin(phi)

    # Arcs
    ridge = vec_image_curve(phi, s, phi_s, e_s, R_0, "x")
    inner = vec_image_curve(phi, s, phi_s, e_s, R_0, "in")
    outter = vec_image_curve(phi, s, phi_s, e_s, R_0, "out")

    xridge = ridge * np.cos(phi)
    yridge = ridge * np.sin(phi)
    xinner = inner * np.cos(phi)
    yinner = inner * np.sin(phi)
    xoutter = outter * np.cos(phi)
    youtter = outter * np.sin(phi)

    xridge = xridge[~np.isnan(xridge)]
    yridge = yridge[~np.isnan(yridge)]
    xinner = xinner[~np.isnan(xinner)]
    yinner = yinner[~np.isnan(yinner)]
    xoutter = xoutter[~np.isnan(xoutter)]
    youtter = youtter[~np.isnan(youtter)]

    if s >= R_0 / np.sqrt(1 - e_s * np.cos(2 * phi_s)):
        # Angles
        phi1 = np.float(phi_1(s, phi_s, e_s, R_0))
        phi2 = np.float(phi_2(s, phi_s, e_s, R_0))
        phi3 = np.float(phi_3(s, phi_s, e_s, R_0))
        phi4 = np.float(phi_4(s, phi_s, e_s, R_0))
        phi5 = np.float(phi_5(s, phi_s, e_s, R_0))
        phi6 = np.float(phi_6(s, phi_s, e_s, R_0))
        phi7 = np.float(phi_7(s, phi_s, e_s, R_0))
        phi8 = np.float(phi_8(s, phi_s, e_s, R_0))

        ridge_1 = image_curve(phi1, s, phi_s, e_s, R_0, "x")
        ridge_2 = image_curve(phi2, s, phi_s, e_s, R_0, "x")
        ridge_3 = image_curve(phi3, s, phi_s, e_s, R_0, "x")
        ridge_4 = image_curve(phi4, s, phi_s, e_s, R_0, "x")
        ridge_5 = image_curve(phi5, s, phi_s, e_s, R_0, "x")
        ridge_6 = image_curve(phi6, s, phi_s, e_s, R_0, "x")
        ridge_7 = image_curve(phi7, s, phi_s, e_s, R_0, "x")
        ridge_8 = image_curve(phi8, s, phi_s, e_s, R_0, "x")

        xridge_1 = ridge_1 * np.cos(phi1)
        yridge_1 = ridge_1 * np.sin(phi1)

        xridge_2 = ridge_2 * np.cos(phi2)
        yridge_2 = ridge_2 * np.sin(phi2)

        xridge_3 = ridge_3 * np.cos(phi3)
        yridge_3 = ridge_3 * np.sin(phi3)

        xridge_4 = ridge_4 * np.cos(phi4)
        yridge_4 = ridge_4 * np.sin(phi4)

        xridge_5 = ridge_5 * np.cos(phi5)
        yridge_5 = ridge_5 * np.sin(phi5)

        xridge_6 = ridge_6 * np.cos(phi6)
        yridge_6 = ridge_6 * np.sin(phi6)

        xridge_7 = ridge_7 * np.cos(phi7)
        yridge_7 = ridge_7 * np.sin(phi7)

        xridge_8 = ridge_8 * np.cos(phi8)
        yridge_8 = ridge_8 * np.sin(phi8)

    # Source Plane:
    ###############

    # Source Location
    xsource, ysource = source_curve(phi, s, phi_s, e_s, R_0)

    # Tangential Pseudo Caustic Curves
    tpccx, tpccy = tangential_pseudo_caustic_curve(phi)

    fig, ax = plt.subplots(1, 2, figsize=(20, 9))

    ax[0].set_title("Source Plane")
    ax[0].plot(xsource, ysource, "-", color="magenta")
    ax[0].plot(tpccx, tpccy, "-.", color="darkcyan")
    ax[0].set_xlabel(r"$y_1$")
    ax[0].set_ylabel(r"$y_2$")
    ax[0].relim()
    ax[0].autoscale_view()
    ax[0].yaxis.set_major_formatter(FormatStrFormatter("$%.1f$"))
    ax[0].xaxis.set_major_formatter(FormatStrFormatter("$%.1f$"))
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(5))
    ax[0].yaxis.set_major_locator(plt.MaxNLocator(5))

    ax[1].set_title("Lens Plane")
    ax[1].plot(tcx, tcy, "--", color="black")
    ax[1].plot(xridge, yridge, "-", color="blue")
    ax[1].plot(xinner, yinner, ".", ms=0.2, color="green")
    ax[1].plot(xoutter, youtter, ".", ms=0.2, color="red")
    ax[1].set_xlabel(r"$x_1$")
    ax[1].set_ylabel(r"$x_2$")
    ax[1].relim()
    ax[1].autoscale_view()
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("$%.1f$"))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter("$%.1f$"))
    ax[1].xaxis.set_major_locator(plt.MaxNLocator(5))
    ax[1].yaxis.set_major_locator(plt.MaxNLocator(5))

    if s >= R_0 / np.sqrt(1 - e_s * np.cos(2 * phi_s)):
        ax[1].plot(xridge_1, yridge_1, ".", ms=10, color="orange")
        ax[1].plot(xridge_2, yridge_2, ".", ms=10, color="gray")
        ax[1].plot(xridge_3, yridge_3, ".", ms=10, color="pink")
        ax[1].plot(xridge_4, yridge_4, ".", ms=10, color="black")
        ax[1].plot(xridge_5, yridge_5, ".", ms=10, color="red")
        ax[1].plot(xridge_6, yridge_6, ".", ms=10, color="cyan")
        ax[1].plot(xridge_7, yridge_7, ".", ms=10, color="brown")
        ax[1].plot(xridge_8, yridge_8, ".", ms=10, color="purple")
        plt.show()

    else:
        plt.show()
