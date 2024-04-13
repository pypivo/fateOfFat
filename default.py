import numpy as np
from numba import jit

import info_about_model as model
from coefficient_controller import CoefficientsController

lambda_ = 1.0
sigma = 0.07

alpha_base =        2.0
beta_base  =        0.02
gamma_base =        1.0

CL_GLN_base=1.0/10.0
CL_CAM_base=1.0/10.0
CL_INS_base=1.0/10.0

t_0_input= 0.0
tau_grid_input = 0.1
INS_check_coeff = 400 # [mmol*s]
tau_grid = 0.01 # [min]
t_0 = 400.0 # [min]

HeartRate = 80.0
k_BMR_Glu_ef = 10**(-2)
k_BMR_AA_ef = 10**(-2)
K_BMR_FFF_ef = 10**(-2)
K_BMR_KB_ef = 10**(-2)      

inv_beta_KB_ef = 1.0/(517.0/1000.0)
inv_beta_Glu_ef = 1.0/(699.0/1000.0)
inv_beta_AA_ef = 1.0/(369.5/1000.0)
inv_beta_FFA_ef = 1.0/(2415.6/1000.0)
inv_beta_Muscle = 1.0/(369.5/1000.0)
inv_beta_GG_m = 1.0/(699.0/1000.0)
inv_beta_TG_a = 1.0/(7246.8/1000.0)
inv_beta_GG_h = 1.0/(699.0/1000.0)

MASS_OF_HUMAN = 70.0
E_day = 1500.0 # [kcal/day]
e_sigma = E_day/(24.0*60.0) #[kcal/min]

beta_KB_ef = 517.0/1000.0 # [kcal/mmol]
beta_Glu_ef = 699.0/1000.0 # [kcal/mmol]
beta_AA_ef = 369.5/1000.0 # [kcal/mmol]
beta_FFA_ef = 2415.6/1000.0 # [kcal/mmol]
beta_Muscle = 369.5/1000.0  # [kcal/mmol]
beta_GG_m = 699.0/1000.0 # [kcal/mmol]
beta_TG_a = 7246.8/1000.0 # [kcal/mmol]
beta_GG_h = 699.0/1000.0 # [kcal/mmol]

Glu_ef_start= E_day/beta_Glu_ef/4
AA_ef_start = E_day/beta_AA_ef/4
FFA_ef_start = E_day/beta_FFA_ef/4
KB_ef_start = E_day/beta_KB_ef/4

start_point_dict = {
    # Myocyte
    "Muscle_m": 10.0,
    "AA_m": 10.0,
    "GG_m": 10.0,
    "G6_m": 10.0,
    "G3_m": 10.0,
    "Pyr_m": 10.0,
    "Cit_m": 10.0,
    "OAA_m": 10.0,
    "CO2_m": 10.0,
    "H2O_m": 10.0,
    "H_cyt_m": 10.0,
    "H_mit_m": 10.0,
    "Ac_CoA_m": 10.0,
    "FA_CoA_m": 10.0,
    "ATP_cyt_m": 10.0,
    "ATP_mit_m": 10.0,

    # Adipocyte
    "TG_a": 10.0,
    "AA_a": 10.0,
    "G6_a": 10.0,
    "G3_a": 10.0,
    "Pyr_a": 10.0,
    "Ac_CoA_a": 10.0,
    "FA_CoA_a": 10.0,
    "Cit_a": 10.0,
    "OAA_a": 10.0,
    "NADPH_a": 10.0,

    # Hepatocyte
    "GG_h": 10.0,
    "G6_h": 10.0,
    "G3_h": 10.0,
    "TG_h": 10.0,
    "Pyr_h": 10.0,
    "MVA_h": 10.0,
    "OAA_h": 10.0,
    "Cit_h": 10.0,
    "AA_h": 10.0,
    "NADPH_h": 10.0,
    "Ac_CoA_h": 10.0,
    "FA_CoA_h": 10.0,

    # Fluid
    "Urea_ef": 10.0,
    "Glu_ef": Glu_ef_start,
    "AA_ef": AA_ef_start,
    "FFA_ef": FFA_ef_start, 
    "KB_ef": KB_ef_start,
    "Glycerol_ef": 10.0,
    "Lac_m": 10.0,
    "TG_pl": 10.0,
    "Cholesterol_pl": 10.0,    

    # Hormones
    "INS": 0.0,
    "GLN": 0.0,
    "CAM": 0.0,
}

SUBSTANCE_LIST = tuple(name for name in start_point_dict.keys())

velocity_depot = 1.0
power_of_coeff = 10**(-5)
j_base = 0.0


def make_default_coefficients() -> dict:

    coefficients = {}
    for name in model.match_coefficient_name_and_input_substances.keys():
        if name in ['m_1', 'm_3', 'm_4', 'm_5']:
            coefficients[name] = 1
        if name in model.DEPO_COEFFICIENTS:
            coefficients[name] = velocity_depot
        elif "j" in name:
            coefficients[name] = j_base
        else:
            coefficients[name] = power_of_coeff
    coefficients['m_21'] = 10**(-2)
    # print(coefficients)
    

    return coefficients

# DefaultCoefficientsController = make_default_coeff_controller()
# print(DefaultCoefficientsController.adipocyte_coefficients_base)