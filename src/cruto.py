import numpy as np
import numba 
from input import *
import torch
from coefficient_controller import update_coefficients
from local_contributor_config import problem_folder

W =  50.0 # [min] window to check AUC(INS, t-W,t)
INS_check_coeff = 400 # [mmol*s]

# take grid on from FPC.ipynb file 
tau_grid = 0.01 # [min]
t_0 = 400.0 # [min]
t_end = 400.0+1440.0*3 # [min]
# t_end = 500.0 # [min]
t_0_input= 0.0
tau_grid_input = 0.1

N = int((t_end-t_0)/tau_grid)+1
time_grid = np.linspace(start=t_0, stop=t_end, num=N)

# make input data
J_flow_prot_func = torch.load(
    os.path.join(problem_folder, 'ddt_AA_ef'))

J_flow_carb_func = torch.load(
    os.path.join(problem_folder, 'ddt_Glu_ef'))
J_flow_fat_func = torch.load(
    os.path.join(problem_folder, 'ddt_TG_pl'))
J_prot_func = torch.load(
    os.path.join(problem_folder, 'J_prot'))

J_fat_func = torch.load(
    os.path.join(problem_folder, 'J_fat'))
J_carb_func = torch.load(
    os.path.join(problem_folder, 'J_carb'))

beta_KB_ef = 517.0/1000.0 # [kcal/mmol]
beta_Glu_ef = 699.0/1000.0 # [kcal/mmol]
beta_AA_ef = 369.5/1000.0 # [kcal/mmol]
beta_FFA_ef = 2415.6/1000.0 # [kcal/mmol]
beta_Muscle = 369.5/1000.0  # [kcal/mmol]
beta_GG_m = 699.0/1000.0 # [kcal/mmol]
beta_TG_a = 7246.8/1000.0 # [kcal/mmol]
beta_GG_h = 699.0/1000.0 # [kcal/mmol]
# TG_h = ?
# TG_plasma = ?

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

power_of_coeff = -5

k_BMR_Glu_ef = 10**(-2)
k_BMR_AA_ef = 10**(-2)
K_BMR_FFF_ef = 10**(-2)
K_BMR_KB_ef = 10**(-2)



# IF (есть лишние AA) THEN (rest_cont идет на расход AA)
# IF (нет лишних AA AND есть Glu AND есть INS) THEN (rest_cont идет на расход Glu)
# IF (нет инсулина INS AND нет инсулина <= 180 [мин]) THEN (rest_cont идет на расход FFA)
# IF (нет инсулина INS AND нет инсулина > 180 [мин]) THEN (rest_cont идет на расход FFA AND rest_cont идет на расход KB )
# IF (нет инсулина INS AND нет инсулина > 180 [мин]) THEN (рост кетоновых тел v=BMR*(0.5/100.0) [kcal/hour])
# IF (7*60[min] голодания) THEN (расход KB 0.07 + 0.01*7 [kcal/min] v=BMR*(7.0/100.0))
# IF (70*60[min] голодания) THEN (расход KB 0.07 + 0.01*70 [kcal/min]v=BMR*(38.5/100.0))


lambda_ = 1.0
sigma = 0.07

alpha_base       = 2.0
beta_base=         0.02
gamma_base =        1.0

CL_GLN_base=1.0/10.0
CL_CAM_base=1.0/10.0
CL_INS_base=1.0/10.0

# коэффициенты, отвечающие за перекачку энергии из Glu,FFA,KB,AA из крови
m_1_base=            1.0
m_3_base=            1.0
m_4_base=            1.0
m_5_base=            1.0

velocity_depot_power = 1
# расход из депо
# a_7 под вопросом
# h_20 под вопросом
a_3_base=            10.0**(velocity_depot_power)
a_17_base=            10.0**(velocity_depot_power)
a_18_base=            10.0**(velocity_depot_power)
a_19_base=            10.0**(velocity_depot_power)
a_5_base=            10.0**(velocity_depot_power)
a_6_base=            10.0**(velocity_depot_power)
a_8_base=            10.0**(velocity_depot_power)
h_2_base=            10.0**(velocity_depot_power)
h_10_base=            10.0**(velocity_depot_power)
h_12_base=            10.0**(velocity_depot_power)
h_14_base=            10.0**(velocity_depot_power)
h_13_base=            10.0**(velocity_depot_power)
h_15_base=            10.0**(velocity_depot_power)
h_9_base=            10.0**(velocity_depot_power)
m_7_base=            10.0**(velocity_depot_power)
m_8_base=            10.0**(velocity_depot_power)
m_9_base=            10.0**(velocity_depot_power)
m_10_base=           10.0**(velocity_depot_power)

# номера коэффициентов
a_1_base=            10.0**(power_of_coeff)
a_2_base=            10.0**(power_of_coeff)
a_4_base=            10.0**(power_of_coeff)
a_7_base=            10.0**(power_of_coeff)
a_9_base=            10.0**(power_of_coeff)
a_10_base=            10.0**(power_of_coeff)
a_11_base=            10.0**(power_of_coeff)
a_12_base=            10.0**(power_of_coeff)
a_13_base=            10.0**(power_of_coeff)
a_14_base=            10.0**(power_of_coeff)
a_15_base=            10.0**(power_of_coeff)
a_16_base=            10.0**(power_of_coeff)
m_2_base=            10.0**(power_of_coeff)
m_6_base=            10.0**(power_of_coeff)
m_11_base=           10.0**(power_of_coeff)
m_12_base=           10.0**(power_of_coeff)
m_13_base=           10.0**(power_of_coeff)
m_14_base=           10.0**(power_of_coeff)
m_15_base=           10.0**(power_of_coeff)
m_16_base=           10.0**(power_of_coeff) 
m_17_base=           10.0**(power_of_coeff)
m_18_base=           10.0**(power_of_coeff)
m_19_base=           10.0**(power_of_coeff)
m_20_base=           10.0**(power_of_coeff)
m_21_base=           10.0**(-2)
h_1_base=            10.0**(power_of_coeff)
h_3_base=            10.0**(power_of_coeff)
h_4_base=            10.0**(power_of_coeff)
h_5_base=            10.0**(power_of_coeff)
h_6_base=            10.0**(power_of_coeff)
h_7_base=            10.0**(power_of_coeff)
h_8_base=            10.0**(power_of_coeff)
h_11_base=            10.0**(power_of_coeff)
h_16_base=            10.0**(power_of_coeff)
h_17_base=            10.0**(power_of_coeff)
h_18_base=            10.0**(power_of_coeff)
h_19_base=            10.0**(power_of_coeff)
h_20_base=            10.0**(power_of_coeff)
h_21_base=            10.0**(power_of_coeff)
h_22_base=            10.0**(power_of_coeff)
h_23_base=            10.0**(power_of_coeff)
h_24_base=            10.0**(power_of_coeff)
h_25_base=            10.0**(power_of_coeff)
h_26_base=            10.0**(power_of_coeff)
h_27_base=            10.0**(power_of_coeff)
h_28_base=            10.0**(power_of_coeff)
h_29_base=            10.0**(power_of_coeff)

j_0_base = 0.0
j_1_base = 0.0
j_2_base = 0.0
j_3_base = 0.0
j_4_base = 0.0

Glu_ef_start= E_day/beta_Glu_ef/4
AA_ef_start = E_day/beta_AA_ef/4
FFA_ef_start = E_day/beta_FFA_ef/4
KB_ef_start = E_day/beta_KB_ef/4


@jit(nopython = True)
def cruto_vec(
    t: float, y_vec: np.array,
    INS_on_grid:np.array, INS_AUC_w_on_grid:np.array,T_a_on_grid:np.array,
    last_seen_time:np.array, last_time_pos:np.array,
    J_flow_carb_vs:np.array,
    J_flow_prot_vs:np.array,
    J_flow_fat_vs:np.array,

    myocyte_coefficients_base,
    adipocyte_coefficients_base,
    hepatocyte_coefficients_base,
    fluid_coefficients_base,
):
    m_, a_, h_, j_ = update_coefficients(
        substances_concentration=y_vec,
        myocyte_coefficients_base=myocyte_coefficients_base,
        adipocyte_coefficients_base=adipocyte_coefficients_base,
        hepatocyte_coefficients_base=hepatocyte_coefficients_base,
        fluid_coefficients_base=fluid_coefficients_base,
    )


    buffer = np.zeros(shape=(50, ),dtype=np.float32)
    time_index_i = np.intc((t-t_0_input)/tau_grid_input)
    J_carb_flow = J_flow_carb_vs[time_index_i]
    J_prot_flow = J_flow_prot_vs[time_index_i]
    J_fat_flow  = J_flow_fat_vs[time_index_i]
    t_pos = np.maximum(np.intc(0), np.intc((t-t_0)/tau_grid))
    HeartRate = 80.0

    # Y_{t} values
    # значения в момент времени t                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

    # Myocyte
    Muscle_m = y_vec[0]
    AA_m = y_vec[1]
    GG_m = y_vec[2]
    G6_m = y_vec[3]
    G3_m = y_vec[4]
    Pyr_m = y_vec[5]
    Cit_m = y_vec[6]
    OAA_m = y_vec[7]
    CO2_m = y_vec[8]
    H2O_m = y_vec[9]
    H_cyt_m = y_vec[10]
    H_mit_m = y_vec[11]
    Ac_CoA_m = y_vec[12]
    FA_CoA_m = y_vec[13]
    ATP_cyt_m = y_vec[14]
    ATP_mit_m = y_vec[15]

    # Adipocyte
    TG_a = y_vec[16]
    AA_a = y_vec[17]
    G6_a = y_vec[18]
    G3_a = y_vec[19]
    Pyr_a = y_vec[20]
    Ac_CoA_a = y_vec[21]
    FA_CoA_a = y_vec[22]
    Cit_a = y_vec[23]
    OAA_a = y_vec[24]
    NADPH_a = y_vec[25]

    # Hepatocyte
    GG_h = y_vec[26]
    G6_h = y_vec[27]
    G3_h = y_vec[28]
    TG_h = y_vec[29]
    Pyr_h = y_vec[30]
    MVA_h = y_vec[31]
    OAA_h = y_vec[32]
    Cit_h = y_vec[33]
    AA_h = y_vec[34]
    NADPH_h = y_vec[35]
    Ac_CoA_h = y_vec[36]
    FA_CoA_h = y_vec[37]

    # Fluid
    Urea_ef = y_vec[38]
    Glu_ef = y_vec[39]
    AA_ef = y_vec[40]
    FFA_ef = y_vec[41]
    KB_ef = y_vec[42]
    Glycerol_ef = y_vec[43]
    Lac_m = y_vec[44]
    TG_pl = y_vec[45]
    Cholesterol_pl = y_vec[46]

    # Hormones
    INS = y_vec[47]
    GLN = y_vec[48]
    CAM = y_vec[49]               

    # AUC 

    AUC_at_t = -1.0
    T_a_t = -1.0

    if t_pos - last_time_pos[0] > 0:
        diff_ = t_pos - last_time_pos[0]
        T_a_current = T_a_on_grid[last_time_pos[0]]
        last_seen_time[0] = t 
        t_minus_w_pos = np.maximum(np.intc(0), np.intc((t-W-t_0)/tau_grid))

        for j in range(1,diff_+1):
            INS_on_grid[last_time_pos[0]+j] = INS

        AUC_at_t = AUC_at_linear_grid(tau_grid, INS_on_grid, t_minus_w_pos, t_pos)

        for j in range(1,diff_+1):
            INS_AUC_w_on_grid[last_time_pos[0]+j] = AUC_at_t

        if AUC_at_t < INS_check_coeff and (t-t_0) >= W:
            for j in range(1,diff_+1):
                T_a_on_grid[last_time_pos[0]+j] = T_a_current + tau_grid*j
        else:
            for j in range(1,diff_+1):
                T_a_on_grid[last_time_pos[0]+j] = 0.0
        T_a_t = T_a_current + tau_grid*diff_
        last_time_pos[0] += diff_
    else:
        # already seen time point. get AUC and T_{a}
        AUC_at_t = INS_AUC_w_on_grid[t_pos]
        T_a_t = T_a_on_grid[t_pos]

    # BMR
    e_KB_plus = 0.005*e_sigma*Heviside(T_a_t-180.0)
    J_KB_plus = e_KB_plus*inv_beta_KB_ef*Heviside(T_a_t-180.0)
    J_Glu_minus = k_BMR_Glu_ef*Glu_ef
    J_AA_minus = k_BMR_AA_ef*AA_ef
    J_FFA_minus = K_BMR_FFF_ef*FFA_ef
    J_KB_minus = K_BMR_KB_ef*KB_ef

    alpha = alpha_base
    beta = beta_base
    gamma = gamma_base
    CL_INS = CL_INS_base
    CL_GLN = CL_GLN_base
    CL_CAM = CL_CAM_base

    # 2. Myocyte
    M_1 = m_[1] * Glu_ef
    M_2 = m_[2] * Pyr_m * H_cyt_m
    M_3 = m_[3] * KB_ef
    M_4 = m_[4] * FFA_ef
    M_5 = m_[5] * AA_ef
    M_6 = m_[6] * AA_m
    M_7 = m_[7] * G6_m
    M_8 = m_[8] * GG_m
    M_9 = m_[9] * G6_m
    M_10 = m_[10] * G3_m
    M_11 = m_[11] * Pyr_m
    M_12 = m_[12] * FA_CoA_m
    M_13 = m_[13] * Ac_CoA_m * OAA_m
    M_14 = m_[14] * Cit_m
    M_15 = m_[15] * H_cyt_m
    M_16 = m_[16] * H_mit_m # * [O2]
    M_17 = m_[17] * AA_m
    M_18 = m_[18] * AA_m
    M_19 = m_[19] * AA_m
    M_20 = m_[20] * AA_m
    M_21 = m_[21] * Muscle_m
    #3. Adipocyte
    A_1 = a_[1] * AA_ef
    A_2 = a_[2] * FFA_ef
    A_3 = a_[3] * TG_a
    A_4 = a_[4] * Glu_ef
    A_5 = a_[5] * G6_a
    A_6 = a_[6] * G6_a
    A_7 = a_[7] * G3_a * FA_CoA_a
    A_8 = a_[8] * G3_a
    A_9 = a_[9] * OAA_a
    A_10 = a_[10] * Pyr_a
    A_11 = a_[11] * Pyr_a
    A_12 = a_[12] * OAA_a
    A_13 = a_[13] * Ac_CoA_a * NADPH_a
    A_14 = a_[14] * Cit_a
    A_15 = a_[15] * Cit_a
    A_16 = a_[16] * OAA_a * Ac_CoA_a
    A_17 = a_[17] * AA_a
    A_18 = a_[18] * AA_a
    A_19 = a_[19] * AA_a
    #4. Hepatocyte
    H_1 = h_[1] * AA_ef
    H_2 = h_[2] * G6_h
    H_3 = h_[3] * Glu_ef
    H_4 = h_[4] * Glycerol_ef
    H_5 = h_[5] * Lac_m
    H_6 = h_[6] * Ac_CoA_h
    H_7 = h_[7] * MVA_h
    H_8 = h_[8] * FFA_ef
    H_9 = h_[9] * TG_h
    H_10 = h_[10] * G6_h
    H_11 = h_[11] * GG_h
    H_12 = h_[12] * G6_h
    H_13 = h_[13] * G3_h
    H_14 = h_[14] * G6_h
    H_15 = h_[15] * G3_h
    H_16 = h_[16] * Pyr_h
    H_17 = h_[17] * Ac_CoA_h * NADPH_h
    H_18 = h_[18] * FA_CoA_h
    H_19 = h_[19] * Ac_CoA_h * NADPH_h
    H_20 = h_[20] * G3_h * FA_CoA_h
    H_21 = h_[21] * Ac_CoA_h * OAA_h
    H_22 = h_[22] * Cit_h
    H_23 = h_[23] * OAA_h
    H_24 = h_[24] * OAA_h
    H_25 = h_[25] * Pyr_h
    H_26 = h_[26] * Cit_h
    H_27 = h_[27] * AA_h
    H_28 = h_[28] * AA_h
    H_29 = h_[29] * AA_h

    J_0 = j_[0] * TG_pl
    J_1 = j_[1] * Glu_ef
    J_2 = j_[2] * KB_ef
    J_3 = j_[3] * FFA_ef
    J_4 = j_[4] * AA_ef
    
    # вычисление вектора F(t) в точке t
    # депо
    right_TG_a =2.0*A_7 - A_3
    right_AA_a =A_1 - A_17 - A_18 - A_19 
    right_G6_a =A_4 - A_5 - A_6
    right_G3_a =A_5 + (1.0/2.0)*A_6 + A_9 - A_7 - A_8
    right_GG_h = H_10 - H_11
    right_G6_h = H_3 + H_11 + H_13 - H_2 - H_10 - H_12 - H_14
    right_G3_h = H_4 + H_12 + (1.0/2.0)*H_14 + H_23 - H_13 - H_15 - H_20
    right_TG_h = 2.0*H_20 - H_9
    right_GG_m = M_7 - M_8
    right_G6_m = M_1 + M_8 - M_7 - M_9
    right_G3_m = M_9 - M_10
    right_TG_pl =  J_fat_flow + H_9 - J_0

    right_Pyr_a =A_8 + (1.0/2.0)*A_12 + (1.0/2.0)*A_19 - A_10 - A_11
    right_Ac_CoA_a =A_10 + (1.0/2.0)*A_14 + (1.0/2.0)*A_18 - A_13 - A_16
    right_FA_CoA_a =A_2 + 2.0*A_13 - A_7
    right_Cit_a =2.0*A_16 - A_14 - A_15
    right_OAA_a =A_11 + (1.0/2.0)*A_14 + A_15 +(1.0/2.0)*A_17 - A_9 - A_12 - A_16 
    right_NADPH_a =(1.0/2.0)*A_6 + (1.0/2.0)*A_12 - A_13

    right_Pyr_h =    H_5 + H_15 + (1.0/2.0)*H_24 + (1.0/2.0)*H_29 - H_16 - H_25
    right_Ac_CoA_h = H_16 + H_18 + H_26 + (1.0/2.0)*H_27 - H_17 - H_19 - H_6
    right_FA_CoA_h = H_8 + 2.0*H_19 - H_18 - H_20
    right_MVA_h =    H_17 - H_7
    right_OAA_h =    H_22 + H_25 + H_26  + (1.0/2.0)*H_28 - H_21 - H_23 - H_24
    right_Cit_h =    H_21 - H_22 - H_26 
    right_AA_h = H_1 - H_27 - H_28 - H_29
    right_NADPH_h =  (1.0/2.0)*H_14 + (1.0/2.0)*H_24 - H_19
    
    right_Pyr_m =    (1.0/3.0)*M_10 + (1.0/2.0)*M_17 - M_11 - M_2
    right_Ac_CoA_m = (1.0/2.0)*M_3 + (1.0/2.0)*M_11 + (1.0/2.0)*M_12 + (1.0/2.0)*M_18 - M_13
    right_FA_CoA_m = M_4 - M_12
    right_AA_m = M_5 + M_21 - M_6 - M_17 - M_18 - M_19 - M_20
    right_Cit_m =    2.0*M_13 - M_14
    right_OAA_m =    (1.0/2.0)*M_14 + (1.0/2.0)*M_19 - M_13
    right_H_cyt_m =  (1.0/3.0)*M_10  - M_15 - M_2
    right_H_mit_m =  (1.0/2.0)*M_3 + (1.0/2.0)*M_12 + M_15 - M_16
    right_CO2_m =    (1.0/2.0)*M_11 + (1.0/2.0)*M_14
    right_H2O_m =    (1.0/2.0)*M_16
    right_ATP_cyt_m =    (1.0/3.0)*M_10 
    right_ATP_mit_m =    (1.0/2.0)*M_16 
    
    right_Glu_ef = J_carb_flow + H_2 - J_Glu_minus - M_1 - A_4 - H_3   - J_1
    right_AA_ef =  J_prot_flow + M_6  - J_AA_minus - M_5 - A_1 - H_1  - J_4 
    right_FFA_ef= J_0 + (1.0/2.0)*A_3  - J_FFA_minus - M_4 - A_2 - H_8 - J_3  
    right_KB_ef=  - J_KB_minus + J_KB_plus - M_3 - J_2 + H_6

    
    right_Glycerol_ef =    J_0 + (1.0/2.0)*A_3 - H_4
    right_Lac_m=  2.0*M_2 - H_5
    right_Urea_ef=    J_4 + (1.0/2.0)*A_17 + (1.0/2.0)*A_18 + (1.0/2.0)*A_19 + (1.0/2.0)*M_17 + (1.0/2.0)*M_18 + (1.0/2.0)*M_19 + (1.0/2.0)*H_27 + (1.0/2.0)*H_28 + (1.0/2.0)*H_29
    right_Cholesterol_pl= H_7
    right_INS =  - INS * CL_INS  +1.0 * J_carb_flow  +1.0 * J_fat_flow + 1.0 * J_prot_flow  # +1.0 * Glu_ef * Heviside((Glu_ef-5.0)/14.0) #
    right_GLN = - CL_GLN * GLN  + lambda_ * (1.0/np.maximum(Glu_ef/14.0, 0.1)) # не химическая кинетика
    right_CAM = sigma * HeartRate - CL_CAM * CAM
    right_Muscle_m = M_20 - M_21

    buffer[0] = right_Muscle_m
    buffer[1] = right_AA_m
    buffer[2] = right_GG_m
    buffer[3] = right_G6_m
    buffer[4] = right_G3_m
    buffer[5] = right_Pyr_m
    buffer[6] = right_Cit_m
    buffer[7] = right_OAA_m
    buffer[8] = right_CO2_m
    buffer[9] = right_H2O_m
    buffer[10] = right_H_cyt_m
    buffer[11] = right_H_mit_m
    buffer[12] = right_Ac_CoA_m
    buffer[13] = right_FA_CoA_m
    buffer[14] = right_ATP_cyt_m
    buffer[15] = right_ATP_mit_m

    buffer[16] = right_TG_a
    buffer[17] = right_AA_a
    buffer[18] = right_G6_a
    buffer[19] = right_G3_a
    buffer[20] = right_Pyr_a
    buffer[21] = right_Ac_CoA_a
    buffer[22] = right_FA_CoA_a
    buffer[23] = right_Cit_a
    buffer[24] = right_OAA_a
    buffer[25] = right_NADPH_a

    buffer[26] = right_GG_h
    buffer[27] = right_G6_h
    buffer[28] = right_G3_h
    buffer[29] = right_TG_h
    buffer[30] = right_Pyr_h
    buffer[31] = right_MVA_h
    buffer[32] = right_OAA_h
    buffer[33] = right_Cit_h
    buffer[34] = right_AA_h
    buffer[35] = right_NADPH_h
    buffer[36] = right_Ac_CoA_h
    buffer[37] = right_FA_CoA_h

    buffer[38] = right_Urea_ef
    buffer[39] = right_Glu_ef
    buffer[40] = right_AA_ef
    buffer[41] = right_FFA_ef
    buffer[42] = right_KB_ef
    buffer[43] = right_Glycerol_ef
    buffer[44] = right_Lac_m
    buffer[45] = right_TG_pl
    buffer[46] = right_Cholesterol_pl

    buffer[47] = right_INS
    buffer[48] = right_GLN
    buffer[49] = right_CAM

    return buffer
