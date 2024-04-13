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
        return processes_coefficients_base

    def update_base_coefficient_value(self, name: str, value: float):
        """
        метод для изменения значения коэффициента(из интерфейса, при прогоне коэфов)
        """
        self.coefficients[name] = value

    def calculate_change_in_inner_processes(self, substances_concentration: list[float]) -> tuple[list[float]]:
        """
        Расчет изменения входящего вещества в реакции: G6_h (h_1)-> Glu_ef
        """
        m, a, h, j = CoefficientsController._calculate_change_in_inner_processes(
            substances_concentration=substances_concentration,
            match_coefficient_and_substances_ind=model.jit_match_coefficient_and_input_substances_ind,
            myocyte_coefficients_base= self.myocyte_coefficients_base,
            adipocyte_coefficients_base= self.adipocyte_coefficients_base,
            hepatocyte_coefficients_base= self.hepatocyte_coefficients_base,
            fluid_coefficients_base= self.fluid_coefficients_base,

            myocyte_processes = model.myocyte_processes,
            adipocyte_processes = model.adipocyte_processes,
            hepatocyte_processes = model.hepatocyte_processes,
            fluid_processes = model.fluid_processes,
        )
        return m, a, h, j

    @staticmethod
    # @jit(nopython = True)
    def _calculate_change_in_inner_processes(
        substances_concentration: list[float],
        match_coefficient_and_substances_ind: dict[str, list[int]],
        myocyte_coefficients_base: list[float],
        adipocyte_coefficients_base: list[float],
        hepatocyte_coefficients_base: list[float],
        fluid_coefficients_base: list[float],

        myocyte_processes: np.array,
        adipocyte_processes: np.array,
        hepatocyte_processes: np.array,
        fluid_processes: np.array,
    ) -> tuple[list[float]]:
        """
        Расчет изменения входящего вещества в реакции: G6_h (h_1)-> Glu_ef
        jit разогнанный
        """

        calculate_change_in_myocyte_processes(
            myocyte_coefficients_base,
            myocyte_processes,
            substances_concentration,
            match_coefficient_and_substances_ind,
        )
        calculate_change_in_adipocyte_processes(
            adipocyte_coefficients_base,
            adipocyte_processes,
            substances_concentration,
            match_coefficient_and_substances_ind,
        )
        calculate_change_in_hepatocyte_processes(
            hepatocyte_coefficients_base,
            hepatocyte_processes,
            substances_concentration,
            match_coefficient_and_substances_ind,
        )
        calculate_change_in_fluid_processes(
            fluid_coefficients_base,
            fluid_processes,
            substances_concentration,
            match_coefficient_and_substances_ind,
        )

        return myocyte_processes, adipocyte_processes, hepatocyte_processes, fluid_processes


@jit(nopython = True)
def calculate_change_in_myocyte_processes(
    myocyte_coefficients_base: list[float],
    myocyte_processes: np.array,
    substances_concentration: list[float],
    match_coefficient_and_substances_ind: dict[str, list[int]],
    myocyte_coefficients_names: tuple[str] = model.myocyte_coefficients_names,
):
    for i in range(1, len(myocyte_processes)):
        name = myocyte_coefficients_names[i]
        current_coefficient = update_coefficients(
            name,
            myocyte_coefficients_base[i],
            substances_concentration,
        )
        
        # k * [A] * [B]
        myocyte_processes[i] = current_coefficient
        for substance_ind in match_coefficient_and_substances_ind[name]:
            myocyte_processes[i] *= substances_concentration[substance_ind]


@jit(nopython = True)
def calculate_change_in_adipocyte_processes(
    adipocyte_coefficients_base: list[float],
    adipocyte_processes: np.array,
    substances_concentration: list[float],
    match_coefficient_and_substances_ind: dict[str, list[int]],
    adipocyte_coefficients_names: tuple[str] = model.adipocyte_coefficients_names,
):
    for i in range(1, len(adipocyte_processes)):
        name = adipocyte_coefficients_names[i]
        current_coefficient = update_coefficients(
            name,
            adipocyte_coefficients_base[i],
            substances_concentration,
        )
        
        # k * [A] * [B]
        adipocyte_processes[i] = current_coefficient
        for substance_ind in match_coefficient_and_substances_ind[name]:
            adipocyte_processes[i] *= substances_concentration[substance_ind]


@jit(nopython = True)
def calculate_change_in_hepatocyte_processes(
    hepatocyte_coefficients_base: list[float],
    hepatocyte_processes: np.array,
    substances_concentration: list[float],
    match_coefficient_and_substances_ind: dict[str, list[int]],
    hepatocyte_coefficients_names: tuple[str] = model.hepatocyte_coefficients_names,
):
    for i in range(1, len(hepatocyte_processes)):
        name = hepatocyte_coefficients_names[i]
        current_coefficient = update_coefficients(
            name,
            hepatocyte_coefficients_base[i],
            substances_concentration,
        )
        
        # k * [A] * [B]
        hepatocyte_processes[i] = current_coefficient
        for substance_ind in match_coefficient_and_substances_ind[name]:
            hepatocyte_processes[i] *= substances_concentration[substance_ind]


@jit(nopython = True)
def calculate_change_in_fluid_processes(
    fluid_coefficients_base: list[float],
    fluid_processes: np.array,
    substances_concentration: list[float],
    match_coefficient_and_substances_ind: dict[str, list[int]],
    fluid_coefficients_names: tuple[str] = model.fluid_coefficients_names,
):
    for i in range(1, len(fluid_processes)):
        name = fluid_coefficients_names[i]
        current_coefficient = update_coefficients(
            name,
            fluid_coefficients_base[i],
            substances_concentration,
        )
        
        # k * [A] * [B]
        fluid_processes[i] = current_coefficient
        for substance_ind in match_coefficient_and_substances_ind[name]:
            fluid_processes[i] *= substances_concentration[substance_ind]  


@jit(nopython = True)
def update_coefficients(
    name: str,
    coefficient: float,
    substances_concentration: list[float],
    insulin_coefficients: tuple[str] = model.INSULIN_COEFFICIENTS,
    glucagon_coefficients: tuple[str] = model.GLUCAGON_COEFFICIENTS,
):
    """
    обновляет значения коэффициентов в зависимости от процесса
    можно настроить по разному, пока что сделан Heviside
    """
    if name in insulin_coefficients:
        return _update_insulin_coefficient(coefficient, substances_concentration)
    elif name in glucagon_coefficients:
        return _update_glucagon_coefficient(coefficient, substances_concentration)
    
    #  для простых коэффициентов просто возвращается значение
    return coefficient


@jit(nopython = True)            
def _update_insulin_coefficient(
    coefficient: float,
    substances_concentration: list[float],
    activation_coefficient: float = model.INS_ACTIVATION,
    INS_i: int = model.INS_i,
):
    """
    обновление инсулиновых коэффициентов
    """
    INS = substances_concentration[INS_i]
    return coefficient * Heviside(INS - activation_coefficient)



@jit(nopython = True)            
def _update_glucagon_coefficient(
    coefficient: float,
    substances_concentration: list[float],
    activation_coefficient: float = model.GLU_ACTIVATION,
    GLN_i: int = model.GLN_i,
    CAM_i: int = model.CAM_i
):
    """
    обновление глюкагеновых коэффициентов
    """
    GLN = substances_concentration[GLN_i]
    CAM = substances_concentration[CAM_i]
    return coefficient * Heviside(GLN  + CAM - activation_coefficient)

# @njit
# def a(c: int):
#     b = c
#     k = 0
#     for i in range(600000000):
#         if b > 5:
#             b -= 1
#             k -= i%10000
#         else:
#             b += 1
#             k += i%10000
#     return b


# @njit
# def b(arr: np.array):
#     b = 0
#     for i in range(6000):
#         # arr = np.zeros(shape=(21 + 1,), dtype=np.float64)
        
#         if b > 5:
#             b -= 1
#             arr[i%10] -= b
#         else:
#             b += 1
#             arr[i%10] += b
#     return b

# @njit
# def wb(arr: np.array):
#     a = 0
#     for i in range(200000):
#         a += b(arr)
#     return a


# print('started')

# # t = time.time()
# # print(a(0))
# # print("standart", time.time() - t)

# t = time.time()
# print(wb(model.processes))
# print(model.processes)
# print("imported", time.time() - t)