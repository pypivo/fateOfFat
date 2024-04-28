from copy import copy
import time


import numpy as np
import numpy.ma as ma
from numba import jit, typed, types, njit

import info_about_model as model

@jit(nopython=True)
def Heviside(x:float) -> float:
    if x < 0.0:
        return 0.0
    return 1.0


class CoefficientsController:
    
    def __init__(self, coefficients: dict[str, float]) -> None:
        self.coefficients = coefficients

        self.myocyte_coefficients_base = self.set_processes_coefficients(model.myocyte_coefficients_names[1:])
        self.adipocyte_coefficients_base = self.set_processes_coefficients(model.adipocyte_coefficients_names[1:])
        self.hepatocyte_coefficients_base = self.set_processes_coefficients(model.hepatocyte_coefficients_names[1:])
        self.fluid_coefficients_base = self.set_processes_coefficients(model.fluid_coefficients_names[1:])

    def set_processes_coefficients(self, processes: list[str]):
        processes_coefficients_base = [0.0]
        for process in processes:
            processes_coefficients_base.append(self.coefficients[process])
        return tuple(processes_coefficients_base)

    def update_base_coefficient_value(self, name: str, value: float):
        """
        метод для изменения значения коэффициента(из интерфейса, при прогоне коэфов)
        """
        self.coefficients[name] = value


@njit
def update_coefficients(
    substances_concentration: list[float],

    myocyte_coefficients_base,
    adipocyte_coefficients_base,
    hepatocyte_coefficients_base,
    fluid_coefficients_base,
) -> tuple[list[float]]:
    """
    Расчет изменения входящего вещества в реакции: G6_h (h_1)-> Glu_ef
    jit разогнанный
    """
    m_ = np.array(myocyte_coefficients_base)
    a_ = np.array(adipocyte_coefficients_base)
    h_ = np.array(hepatocyte_coefficients_base)
    j_ = np.array(fluid_coefficients_base)

    update_insulin_coefficients(
        substances_concentration=substances_concentration,

        m_base=myocyte_coefficients_base,
        a_base=adipocyte_coefficients_base,
        h_base=hepatocyte_coefficients_base,
        j_base=fluid_coefficients_base,

        m_=m_,
        a_=a_,
        h_=h_,
        j_=j_,
    )

    update_glucagon_coefficient(
        substances_concentration=substances_concentration,

        m_base=myocyte_coefficients_base,
        a_base=adipocyte_coefficients_base,
        h_base=hepatocyte_coefficients_base,
        j_base=fluid_coefficients_base,

        m_=m_,
        a_=a_,
        h_=h_,
        j_=j_,
    )

    return m_, a_, h_, j_


@jit(nopython = True)            
def update_insulin_coefficients(
    substances_concentration: list[float],

    m_base: list[float],
    a_base: list[float],
    h_base: list[float],
    j_base: list[float],

    m_: np.array,
    a_: np.array,
    h_: np.array,
    j_: np.array,

    activation_coefficient: float = model.INS_ACTIVATION,
    INS_i: int = model.INS_i,
):
    """
    обновление инсулиновых коэффициентов
    """
    INS = substances_concentration[INS_i]
    insulin_regulation = Heviside(INS - activation_coefficient)

    m_[1] = m_base[1] * insulin_regulation
    m_[7] = m_base[7] * insulin_regulation
    m_[9] = m_base[9] * insulin_regulation
    m_[11] = m_base[11] * insulin_regulation

    a_[2] = a_base[2] * insulin_regulation
    a_[4] = a_base[4] * insulin_regulation
    a_[5] = a_base[5] * insulin_regulation
    a_[7] = a_base[7] * insulin_regulation
    a_[10] = a_base[10] * insulin_regulation
    a_[12] = a_base[12] * insulin_regulation
    a_[13] = a_base[13] * insulin_regulation
    a_[14] = a_base[14] * insulin_regulation

    h_[3] = h_base[3] * insulin_regulation
    h_[7] = h_base[7] * insulin_regulation
    h_[10] = h_base[10] * insulin_regulation
    h_[12] = h_base[12] * insulin_regulation
    h_[16] = h_base[16] * insulin_regulation
    h_[17] = h_base[17] * insulin_regulation
    h_[19] = h_base[19] * insulin_regulation
    h_[20] = h_base[20] * insulin_regulation
    h_[24] = h_base[24] * insulin_regulation
    h_[26] = h_base[26] * insulin_regulation

    j_[0] = j_base[0] * insulin_regulation



@jit(nopython = True)            
def update_glucagon_coefficient(
    substances_concentration: list[float],

    m_base: list[float],
    a_base: list[float],
    h_base: list[float],
    j_base: list[float],

    m_: np.array,
    a_: np.array,
    h_: np.array,
    j_: np.array,

    activation_coefficient: float = model.GLU_ACTIVATION,
    GLN_i: int = model.GLN_i,
    CAM_i: int = model.CAM_i
):
    """
    обновление глюкагеновых коэффициентов
    """
    GLN = substances_concentration[GLN_i]
    CAM = substances_concentration[CAM_i]
    glucagon_regulation = Heviside(GLN  + CAM - activation_coefficient)

    m_[8] = m_base[8] * glucagon_regulation

    a_[3] = a_base[3] * glucagon_regulation
    a_[9] = a_base[9] * glucagon_regulation
    a_[11] = a_base[11] * glucagon_regulation

    h_[2 ]=  h_base[2] * glucagon_regulation
    h_[6 ]=  h_base[6] * glucagon_regulation
    h_[11] = h_base[11] * glucagon_regulation 
    h_[13] = h_base[13] * glucagon_regulation 
    h_[18] = h_base[18] * glucagon_regulation 
    h_[23] = h_base[23] * glucagon_regulation 
    h_[25] = h_base[25] * glucagon_regulation 
