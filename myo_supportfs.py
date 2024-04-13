import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from numba import jit
from copy import deepcopy
from tqdm import tqdm
import plotly
import plotly.graph_objects as go
# def F_carb(x): # [g]
#     return np.exp(-1.0*x)*10 * np.abs(np.cos(2*x))
#
# def F_fat(x): # [g]
#     return np.exp(-2.0*x)*10* np.abs(np.cos(2*x))
#
# def F_prot(x): # [g]
#     return np.exp(-3.0*x)*10* np.abs(np.cos(2*x))
#
#
#
# def makeFcarb():
#

def read_diet(path_to_exel):
    diet = pd.read_excel(path_to_exel)
    print([el for el in diet])
    # diet = pd.read_excel(r"C:\Users\User\PycharmProjects\gr_lab_2023\legacy_code\diet_Mikhail.xlsx")

    # Начало приёма пиши
    meal_beginning = np.array(diet.MealTime.dropna())

    # Выбор периода диеты в днях. Атоматическое определение того, сколько недель (даже не целых) занимает диета.
    diet_period = 14  # days
    weeks = math.ceil(diet_period / 7)

    awake_time = 7.5  # Час пробуждения
    assert awake_time < np.min(meal_beginning), 'Error: awake_time should be smaller than the first meal time'

    sleep_time = 23.5  # Час отбоя
    assert sleep_time > np.max(meal_beginning), 'Error: sleep_time should be bigger than the last meal time'

    # Регулирование калорийности диеты в процентах от уже установленной
    calorage = np.array([100, 100])  # ,110,120,130,140])
    assert len(calorage) == weeks, 'Error: lenth of calorage should be equal number of weeks'

    time_for_main_meals = 30  # продолжительность основных приёмов пищи в минутах
    time_for_minor_meals = 10  # продолжительность неосновных приёмов пищи в минутах
    meals_per_day = len(meal_beginning)  # кол-во приёмов пищи в день
    meal_starts = meal_beginning * 60  # время приёма пищи в минутах
    # временная метка подъема [min]. отсчет от начала главного периода.
    t_awake = [awake_time * 60 + i * 24 * 60 for i in range(diet_period)]  # время пробуждения в минутах на каждый день
    # временная метка отбоя [min]. отсчет от начала главного периода.
    t_sleep = [sleep_time * 60 + i * 24 * 60 for i in range(diet_period)]  # время отбоя в минутах на каждый день
    # время старта приема пищи в течении дня. номер стороки - номер дня. номер столбца - номер приема пищи
    t_0 = [[meal_starts[j] + i * 24 * 60 for j in range(meals_per_day)] for i in range(diet_period)]
    # время окончания приема пищи в течении дня. номер стороки - номер дня. номер столбца - номер приема пищи
    t_end = [
        [t_0[i][j] + time_for_main_meals * ((j + 1) % 2) + time_for_minor_meals * (j % 2) for j in range(meals_per_day)]
        for i in range(diet_period)]
    t_0 = np.reshape(t_0, np.size(t_0))  # время начала приёма пищи на каждый день
    t_end = np.reshape(t_end, np.size(
        t_end))  # время конца приёма пищи на каждый день ( 30 минут на обычный приём и 10 на перекус)

    # Чтение данных из таблицы
    Carbs0_, Prots0_, Fats0_, Calories0_, GL_, IL_, GI_, II_ = np.zeros((8, int(np.max(diet.N))))
    for i in range(int(np.max(diet.N))):
        # i - номер дня. в столбце "N" он может дублирвоаться. т.к. за день несколько приемов пищи. а каждая запись-строка
        # - это очередной примем пищи

        # суммарное число углеводов за iй день
        Carbs0_[i] = np.sum([np.array(diet.Carbs[j]) for j in np.where(np.array(diet.N) == (i + 1))])
        # суммарное число белков за iй день
        Prots0_[i] = np.sum([np.array(diet.Protein[j]) for j in np.where(np.array(diet.N) == (i + 1))])
        # суммарное число жиров за iй день
        Fats0_[i] = np.sum([np.array(diet.Fats[j]) for j in np.where(np.array(diet.N) == (i + 1))])
        # суммарное число калорий за iй день
        Calories0_[i] = np.sum([np.array(diet.Calories[j]) for j in np.where(np.array(diet.N) == (i + 1))])
        GL_[i] = np.sum([np.array(diet.Carbs[j]) * np.array(diet.GI[j]) for j in np.where(np.array(diet.N) == (i + 1))])
        IL_[i] = np.sum(
            [np.array(diet.Calories[j]) * np.array(diet.II[j]) for j in np.where(np.array(diet.N) == (i + 1))])
        # доп фильтр, защищающий от ошибок в данных
        if Carbs0_[i] == 0:
            GI_[i] = 0
        else:
            GI_[i] = GL_[i] / Carbs0_[i]
        if Calories0_[i] == 0:
            II_[i] = 0
        else:
            II_[i] = IL_[i] / Calories0_[i]
    # скалирование прочитанных данных в прцоентах(для игры со значениями?)
    # заполнение начальных условий на весь период диеты
    Carbs0, Prots0, Fats0, Calories0, GL, IL, GI, II = np.zeros((8, diet_period * meals_per_day))
    for i in range(diet_period * meals_per_day):
        Carbs0[i] = Carbs0_[i % len(Carbs0_)] * calorage[math.ceil((i + 1) / len(Carbs0_)) - 1] / 100
        Prots0[i] = Prots0_[i % len(Carbs0_)] * calorage[math.ceil((i + 1) / len(Carbs0_)) - 1] / 100
        Fats0[i] = Fats0_[i % len(Carbs0_)] * calorage[math.ceil((i + 1) / len(Carbs0_)) - 1] / 100
        Calories0[i] = Calories0_[i % len(Carbs0_)] * calorage[math.ceil((i + 1) / len(Carbs0_)) - 1] / 100
        GL[i] = GL_[i % len(Carbs0_)] * calorage[math.ceil((i + 1) / len(Carbs0_)) - 1] / 100
        IL[i] = IL_[i % len(Carbs0_)] * calorage[math.ceil((i + 1) / len(Carbs0_)) - 1] / 100
        GI[i] = GI_[i % len(Carbs0_)]
        II[i] = II_[i % len(Carbs0_)]
    return {
        "t_0": t_0,
        "t_end": t_end,
        "Carbs0": Carbs0,
        "Prots0": Prots0,
        "Fats0": Fats0,
        "Calories0": Calories0,
        "GL": GL,
        "IL": IL,
        "GI": GI,
        "II": II
    }


# t is parametr.rest is hidden parameters
# Значение i-го пика. пик соответсвует i-му приему пищи

@jit(nopython=True, cache=True)
def pik(t: float, t_0: float, t_end: float, A: float):
    if t <= t_0 or t > t_end:
        f = 0
    elif t > t_0 and t <= (t_0 + t_end) / 2:
        f = 2 * A * (t - t_0) / (t_end - t_0)
    elif t > (t_0 + t_end) / 2 and t <= t_end:
        f = 2 * A * (t_end - t) / (t_end - t_0)
    return f


# t is parametr.rest is hidden parameters
# значение F_i(t)
@jit(nopython=True, cache=True)
def piki(t: float, t_0: np.array, t_end: np.array, A: np.array):
    J_in = 0
    for i in range(len(A)):
        J_in = J_in + pik(t, t_0[i], t_end[i], A[i])
    return J_in
# t_0 = diet_data["t_0"]
# t_end = diet_data["t_end"]
# Carbs0 = diet_data["Carbs0"]
# Prots0 = diet_data["Prots0"]
# Fats0 = diet_data["Fats0"]
# Calories0 = diet_data["Calories0"]
# GL = diet_data["GL"]
# IL = diet_data["IL"]
# GI = diet_data["GI"]
# II = diet_data["II"]
class chunck:
    t1:float
    t2:float
    mass:float
    rho: float
    def __init__(self,t1,t2,mass) -> None:
        self.t1=t1
        self.t2=t2
        self.mass=mass
        self.rho = mass / (t2-t1)

def make_Fcarb(diet_data: Dict[str, np.array]):
    t_0 = diet_data["t_0"]
    t_end = diet_data["t_end"]
    Carbs0 = diet_data["Carbs0"]
    chunks = [chunck(t1=t_0[i],t2=t_end[i],mass=Carbs0[i]) for i in range(len(t_0))]
    def out(t:float):
        for i in range(len(t_0)):
            t1 = t_0[i]
            t2 = t_end[i]
            mass = Carbs0[i]
            if t>= t1 and t<=t2:
                return mass/(t2-t1)
        return 0.0
    return chunks,out

def make_Fprot(diet_data):

    t_0 = diet_data["t_0"]
    t_end = diet_data["t_end"]
    Prots0 = diet_data["Prots0"]
    chunks = [chunck(t1=t_0[i],t2=t_end[i],mass=Prots0[i]) for i in range(len(t_0))]
    def out(t:float):
        for i in range(len(t_0)):
            t1 = t_0[i]
            t2 = t_end[i]
            mass = Prots0[i]
            if t>= t1 and t<=t2:
                return mass/(t2-t1)
        return 0.0
    return chunks,out


def make_Ffat(diet_data):
    t_0 = diet_data["t_0"]
    t_end = diet_data["t_end"]
    Fats0 = diet_data["Fats0"]
    chunks = [chunck(t1=t_0[i],t2=t_end[i],mass=Fats0[i]) for i in range(len(t_0))]
    def out(t:float):
        for i in range(len(t_0)):
            t1 = t_0[i]
            t2 = t_end[i]
            mass = Fats0[i]
            if t>= t1 and t<=t2:
                return mass/(t2-t1)
        return 0.0
    return chunks,out


def get_start_point_names_mapping(dict_of_start_points):
    index_by_name = {}
    name_by_index = {}
    start_point = np.zeros(shape=(len(dict_of_start_points,)),dtype=np.float32)
    i_ = 0
    for Y_name,Y_0 in dict_of_start_points.items():
        start_point[i_] = Y_0
        index_by_name.update({Y_name:i_})
        name_by_index.update({i_:Y_name})
        i_+=1
    return index_by_name, name_by_index, start_point

@jit(nopython=True)
def Heviside(x:float) -> float:
    if x < 0.0:
        return 0.0
    return 1.0

def get_intervals_of_processes(solutions, time_grid, index_by_name):
    processes = {
        'time_point':time_grid,
        'INS': np.zeros(shape=(len(time_grid),),dtype=np.intc),
        'GLN_CAM': np.zeros(shape=(len(time_grid),),dtype=np.intc),
        'GLN_INS_CAM': np.zeros(shape=(len(time_grid),),dtype=np.intc),
        'fasting':[]
    }
    for i in range(len(time_grid)):
        t_i = time_grid[i]
        INS = solutions[i, index_by_name['INS']]
        GLN = solutions[i, index_by_name['GLN']] 
        CAM = solutions[i, index_by_name['CAM']] 
        insulin_activation_coefficient =  17.0
        is_insulin_process = Heviside(INS-insulin_activation_coefficient)
        glucagon_adrenalin_insulin_activation_coefficient = INS/(GLN+CAM)
        is_glucagon_adrenalin_insulin_process = Heviside(glucagon_adrenalin_insulin_activation_coefficient-1.0)
        is_glucagon_adrenalin_process = 1-int(is_glucagon_adrenalin_insulin_process)



        processes['GLN_CAM'][i]=int(is_glucagon_adrenalin_process)
        processes['GLN_INS_CAM'][i]=int(is_glucagon_adrenalin_insulin_process)
        processes['INS'][i]=int(is_insulin_process)
        # processes['fasting'][int(is_fasting)]
        

    time_vec = processes['time_point']
    intervals_of_active_process = {}
    for ProcessName, Values in processes.items():
        if ProcessName == 'time_point':
            continue
        if len(Values) ==0 :
            intervals_of_active_process.update({ProcessName: []})
            continue
        intervals = []
        current_value = 0
        for i in range(len(time_vec)):
            v_i = Values[i]
            t_i = time_vec[i]
            if current_value == 1 and v_i == 1:
                # отрезок продолжается
                continue
            elif current_value == 1 and v_i ==0:
                # отрезок закончился
                current_value = 0
                intervals[-1].append(time_vec[i-1])
            elif current_value == 0 and v_i ==1:
                # отрезок начался
                intervals.append([])
                intervals[-1].append(t_i)
                current_value = 1
            elif current_value == 0 and v_i == 0:
                # отрезок не начался
                continue 
        if current_value==1 and Values[-1] == 1:
            intervals[-1].append(time_vec[-1])
        intervals_of_active_process.update({ProcessName: intervals})
    return intervals_of_active_process

def plot_intervals_to_plotly_fig(fig, intervals_dict, ProcessNameHighsOfLines, ProcessColor):
    for ProcessName, intervals in intervals_dict.items():
        for i in range(len(intervals)):
            interval = intervals[i]
            y_1 = ProcessNameHighsOfLines[ProcessName]
            y_2 = y_1
            x_1 = interval[0]
            x_2 =  interval[1]
            fig.add_trace(go.Scatter(x=[x_1,x_2],
                                    y=[y_1,y_2],
                                    name=ProcessName,
                                    fill=None,
                                    line=dict(width=4 ,color=ProcessColor[ProcessName])
                                    )
                        )
            fig.add_vline(x=x_1,line_color= "#FFFFFF",line_dash="dash")
            fig.add_vline(x=x_2,line_color= "#FFFFFF",line_dash="dash")
            
            
    return fig  

