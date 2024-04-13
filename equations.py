from copy import copy
from typing import Dict

import numpy as np
import numpy.ma as ma
from numba import jit, prange, types, typed

# from default import DefaultCoefficient
import info_about_model as model
import default as d

@jit(nopython=True)
def Heviside(x:float) -> float:
    if x < 0.0:
        return 0.0
    return 1.0

class EquationsController:

    def __init__(self, coefficient_controller) -> None:
        self.coefficient_controller = coefficient_controller

    def calculate_equations(
            self, 
            t:float,
            substances_concentration: list[float],
            INS_on_grid:np.array, INS_AUC_w_on_grid:np.array,T_a_on_grid:np.array,
            last_seen_time:np.array,last_time_pos:np.array,
            J_flow_carb_vs:np.array,
            J_flow_prot_vs:np.array,
            J_flow_fat_vs:np.array,
        ):

        m_, a_, h_, j_ = self.coefficient_controller.calculate_change_in_inner_processes(
            substances_concentration=substances_concentration
        )
        calculated_substance = self._calculate_equations(
            m_, a_, h_, j_,
            substances_concentration,
            t,
            INS_on_grid, INS_AUC_w_on_grid,T_a_on_grid,
            last_seen_time,last_time_pos,
            J_flow_carb_vs,
            J_flow_prot_vs,
            J_flow_fat_vs,
            model.calculated_substance
        )
        return calculated_substance

    @staticmethod
    @jit(nopython = True)
    def _calculate_equations(
        m_: list[float], a_: list[float], h_: list[float], j_: list[float],
        substances_concentration: list[float],
        t:float,
        INS_on_grid:np.array, INS_AUC_w_on_grid:np.array,T_a_on_grid:np.array,
        last_seen_time:np.array,last_time_pos:np.array,
        J_flow_carb_vs:np.array,
        J_flow_prot_vs:np.array,
        J_flow_fat_vs:np.array,
        calculated_substance:np.array, 
    ) -> dict[str, float]:
        """
        расчет изменений веществ на момент времени t
        """
        # calculated_substance = np.zeros(shape=(len(substances_concentration), ),dtype=np.float64)

        time_index_i = np.intc((t-d.t_0_input)/d.tau_grid_input)
        J_carb_flow = J_flow_carb_vs[time_index_i]
        J_prot_flow = J_flow_prot_vs[time_index_i]
        J_fat_flow  = J_flow_fat_vs[time_index_i]

        T_a_t = calculate_something(
            t, 
            INS_on_grid, INS_AUC_w_on_grid,
            T_a_on_grid,last_seen_time,last_time_pos, 
            INS=substances_concentration[model.hormones_shift + 0]
        )

        calculate_myosyte(calculated_substance=calculated_substance, m_=m_)
        calculate_adipocyte(calculated_substance=calculated_substance, a_=a_)
        calculate_hepatocyte(calculated_substance=calculated_substance, h_=h_)
        calculate_extracellular_fluid(
            substances_concentration=substances_concentration,
            calculated_substance=calculated_substance,
            m_=m_, a_=a_, h_=h_, j_=j_,
            J_carb_flow=J_carb_flow, J_prot_flow=J_prot_flow, J_fat_flow=J_fat_flow,
            T_a_t=T_a_t
        )
        calculate_hormones(
            substances_concentration,
            calculated_substance,
            J_carb_flow=J_carb_flow, J_prot_flow=J_prot_flow, J_fat_flow=J_fat_flow,
            Glu_ef=substances_concentration[model.fluid_shift + 1]
        )

        return calculated_substance
    

@jit(nopython = True)
def calculate_myosyte(
    calculated_substance: dict[str, float],
    m_: list[float],  # change_of_input_substance
    shift: int = model.myocyte_shift, 
):
    calculated_substance[shift + 0] = m_[20] - m_[21]                                                                       # = Muscle_m
    calculated_substance[shift + 1] = m_[5] + m_[21] - m_[6] - m_[17] - m_[18] - m_[19] - m_[20]                            # = AA_m
    calculated_substance[shift + 2] = m_[7] - m_[8]                                                                         # = GG_m
    calculated_substance[shift + 3] = m_[1] + m_[8] - m_[7] - m_[9]                                                         # = G6_m
    calculated_substance[shift + 4] = m_[9] - m_[10]                                                                        # = G3_m
    calculated_substance[shift + 5] = (1.0/3.0)*m_[10] + (1.0/2.0)*m_[17] - m_[11] - m_[2]                                  # = Pyr_m
    calculated_substance[shift + 6] = 2.0*m_[13] - m_[14]                                                                   # = Cit_m
    calculated_substance[shift + 7] = (1.0/2.0)*m_[14] + (1.0/2.0)*m_[19] - m_[13]                                          # = OAA_m
    calculated_substance[shift + 8] = (1.0/2.0)*m_[11] + (1.0/2.0)*m_[14]                                                   # = CO2_m
    calculated_substance[shift + 9] = (1.0/2.0)*m_[16]                                                                      # = H2O_m
    calculated_substance[shift + 10] = (1.0/3.0)*m_[10]  - m_[15] - m_[2]                                                   # = H_cyt_m
    calculated_substance[shift + 11] = (1.0/2.0)*m_[3] + (1.0/2.0)*m_[12] + m_[15] - m_[16]                                 # = H_mit_m
    calculated_substance[shift + 12] = (1.0/2.0)*m_[3] + (1.0/2.0)*m_[11] + (1.0/2.0)*m_[12] + (1.0/2.0)*m_[18] - m_[13]    # = Ac_CoA_m
    calculated_substance[shift + 13] = m_[4] - m_[12]                                                                       # = FA_CoA_m
    calculated_substance[shift + 14] = (1.0/3.0)*m_[10]                                                                     # = ATP_cyt_m
    calculated_substance[shift + 15] = (1.0/2.0)*m_[16]                                                                     # = ATP_mit_m

    
@jit(nopython = True)
def calculate_adipocyte(
    calculated_substance: list[float],
    a_: list[float],  # change_of_input_substance 
    shift: int = model.adipocyte_shift,
):
    calculated_substance[shift + 0] = 2.0*a_[7] - a_[3]                                                              #  = TG_a
    calculated_substance[shift + 1] = a_[1] - a_[17] - a_[18] - a_[19]                                               #  = AA_a
    calculated_substance[shift + 2] = a_[4] - a_[5] - a_[6]                                                          #  = G6_a
    calculated_substance[shift + 3] = a_[5] + (1.0/2.0)*a_[6] + a_[9] - a_[7] - a_[8]                                #  = G3_a
    calculated_substance[shift + 4] = a_[8] + (1.0/2.0)*a_[12] + (1.0/2.0)*a_[19] - a_[10] - a_[11]                  #  = Pyr_a
    calculated_substance[shift + 5] = a_[10] + (1.0/2.0)*a_[14] + (1.0/2.0)*a_[18] - a_[13]- a_[16]                  #  = Ac_CoA_a
    calculated_substance[shift + 6] = a_[2] + 2.0*a_[13] - a_[7]                                                     #  = FA_CoA_a
    calculated_substance[shift + 7] = 2.0*a_[16] - a_[14] - a_[15]                                                   #  = Cit_a
    calculated_substance[shift + 8] = a_[11] + (1.0/2.0)*a_[14] + a_[15] +(1.0/2.0)*a_[17] - a_[9]- a_[12] - a_[16]  #  = OAA_a
    calculated_substance[shift + 9] = (1.0/2.0)*a_[6] + (1.0/2.0)*a_[12] - a_[13]                                    #  = NADPH_a


@jit(nopython = True)
def calculate_hepatocyte(
    calculated_substance: list[float],
    h_: list[float],  # change_of_input_substance
    shift: int = model.hepatocyte_shift,
):
    calculated_substance[shift + 0] = h_[10] - h_[11]                                                            # = GG_h
    calculated_substance[shift + 1] = h_[3] + h_[11] + h_[13] - h_[2]- h_[10] - h_[12] - h_[14]                  # = G6_h
    calculated_substance[shift + 2] = h_[4] + h_[12] + (1.0/2.0)*h_[14] + h_[23] - h_[13] - h_[15] - h_[20]      # = G3_h
    calculated_substance[shift + 3] = 2.0*h_[20] - h_[9]                                                         # = TG_h
    calculated_substance[shift + 4] = h_[5] + h_[15] + (1.0/2.0)*h_[24] + (1.0/2.0)*h_[29] - h_[16] - h_[25]     # = Pyr_h
    calculated_substance[shift + 5] = h_[17] - h_[7]                                                             # = MVA_h
    calculated_substance[shift + 6] = h_[22] + h_[25] + h_[26]  + (1.0/2.0)*h_[28] - h_[21] - h_[23] - h_[24]    # = OAA_h
    calculated_substance[shift + 7] = h_[21] - h_[22] - h_[26]                                                   # = Cit_h
    calculated_substance[shift + 8] = h_[1] - h_[27] - h_[28] - h_[29]                                           # = AA_h
    calculated_substance[shift + 9] =  (1.0/2.0)*h_[14] + (1.0/2.0)*h_[24] - h_[19]                              # = NADPH_h
    calculated_substance[shift + 10] = h_[16] + h_[18] + h_[26] + (1.0/2.0)*h_[27] - h_[17] - h_[19] - h_[6]     # = Ac_CoA_h
    calculated_substance[shift + 11] = h_[8] + 2.0*h_[19] - h_[18] - h_[20]                                      # = FA_CoA_h


@jit(nopython = True)
def calculate_extracellular_fluid(
    substances_concentration: list[float],
    calculated_substance: list[float],
    m_: list[float], a_: list[float], h_: list[float], j_: list[float],
    J_carb_flow: float, J_prot_flow: float, J_fat_flow: float,
    T_a_t: float,
    shift: int = model.fluid_shift,
):
    #  = Urea_ef
    calculated_substance[shift + 0]= j_[4] + (1.0/2.0)*a_[17] + (1.0/2.0)*a_[18] + (1.0/2.0)*a_[19] + (1.0/2.0)*m_[17] + (1.0/2.0)*m_[18] + (1.0/2.0)*m_[19] + (1.0/2.0)*h_[27] + (1.0/2.0)*h_[28] + (1.0/2.0)*h_[29]
    
    J_Glu_minus = d.k_BMR_Glu_ef*substances_concentration[shift + 1]
    J_AA_minus = d.k_BMR_AA_ef*substances_concentration[shift + 2]
    J_FFA_minus = d.K_BMR_FFF_ef*substances_concentration[shift + 3]
    J_KB_minus = d.K_BMR_KB_ef*substances_concentration[shift + 4]
    e_KB_plus = 0.005*d.e_sigma*Heviside(T_a_t-180.0)
    J_KB_plus = e_KB_plus*d.inv_beta_KB_ef*Heviside(T_a_t-180.0)

    calculated_substance[shift + 1] = J_carb_flow + h_[2] - J_Glu_minus - m_[1] - a_[4] - h_[3]   - j_[1]        #  = Glu_ef
    calculated_substance[shift + 2] =  J_prot_flow + m_[6]  - J_AA_minus - m_[5] - a_[1] - h_[1]  - j_[4]        #  = AA_ef
    calculated_substance[shift + 3] = j_[0] + (1.0/2.0)*a_[3]  - J_FFA_minus - m_[4] - a_[2] - h_[8] - j_[3]     #  = FFA_ef 
    calculated_substance[shift + 4] =  - J_KB_minus + J_KB_plus - m_[3] - j_[2] + h_[6]                          #  = KB_ef
    calculated_substance[shift + 5] = j_[0] + (1.0/2.0)*a_[3] -h_[4]                                             #  = Glycerol_ef
    calculated_substance[shift + 6] = 2.0*m_[2] - h_[5]                                                          #  = Lac_m
    calculated_substance[shift + 7] =  J_fat_flow + h_[9] - j_[0]                                                #  = TG_pl
    calculated_substance[shift + 8] = h_[7]                                                                      #  = Cholesterol_pl


@jit(nopython = True)
def calculate_hormones(
    substances_concentration: list[float],
    calculated_substance: list[float],
    J_carb_flow: float, J_prot_flow: float, J_fat_flow: float,
    Glu_ef: float,
    shift: int = model.hormones_shift,
):
    # = INS
    calculated_substance[shift + 0] = - substances_concentration[shift + 0] * d.CL_INS_base  + 1.0 * J_carb_flow  +1.0 * J_fat_flow + 1.0 * J_prot_flow  # +1.0 * Glu_ef * Heviside((Glu_ef-5.0)/14.0) #
    
    # = GLN
    calculated_substance[shift + 1] = - d.CL_GLN_base * substances_concentration[shift + 1]  + d.lambda_ * (1.0/np.maximum(Glu_ef/14.0, 0.1)) # не химическая кинетика
    
    # = CAM
    calculated_substance[shift + 2] = d.sigma * d.HeartRate - d.CL_CAM_base * substances_concentration[shift + 2]


@jit(nopython = True)
def calculate_something(
        t: float,
        INS_on_grid:np.array, INS_AUC_w_on_grid:np.array, T_a_on_grid:np.array,
        last_seen_time:np.array,last_time_pos:np.array, INS,
):
    W =  50.0 # [min] window to check AUC(INS, t-W,t)
    t_pos = np.maximum(np.intc(0), np.intc((t-d.t_0)/d.tau_grid))


    if t_pos - last_time_pos[0] > 0:
        diff_ = t_pos - last_time_pos[0]
        T_a_current = T_a_on_grid[last_time_pos[0]]
        last_seen_time[0] = t 
        t_minus_w_pos = np.maximum(np.intc(0), np.intc((t-W-d.t_0)/d.tau_grid))

        for j in range(1,diff_+1):
            INS_on_grid[last_time_pos[0]+j] = INS

        AUC_at_t = AUC_at_linear_grid(d.tau_grid, INS_on_grid, t_minus_w_pos, t_pos)

        for j in range(1,diff_+1):
            INS_AUC_w_on_grid[last_time_pos[0]+j] = AUC_at_t

        if AUC_at_t < d.INS_check_coeff and (t-d.t_0) >= W:
            for j in range(1,diff_+1):
                T_a_on_grid[last_time_pos[0]+j] = T_a_current + d.tau_grid*j
        else:
            for j in range(1,diff_+1):
                T_a_on_grid[last_time_pos[0]+j] = 0.0
        T_a_t = T_a_current + d.tau_grid*diff_
        last_time_pos[0] += diff_
    else:
        # already seen time point. get AUC and T_{a}
        AUC_at_t = INS_AUC_w_on_grid[t_pos]
        T_a_t = T_a_on_grid[t_pos]
    return T_a_t


@jit(nopython = True)
def AUC_at_linear_grid(tau, y, i1,i2):
    s = 0.0
    for j in range(i1+1, i2):
        s += y[j]
    s = 2.0*s
    s = s + y[i1] + y[i2]
    s = s * tau* 0.5
    return s
