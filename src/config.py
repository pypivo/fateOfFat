import os
from local_contributor_config import project_path, problem_folder

myocyte_translators_of_names_path = os.path.join(problem_folder, 'myocyte_translators')
myocyte_map_from_old_y_name_to_new_name_dict_filename = os.path.join(myocyte_translators_of_names_path, 'myocyte_map_from_old_y_name_to_new_name_dict.txt')
myocyte_map_from_old_param_name_to_new_name_dict_filename = os.path.join(myocyte_translators_of_names_path, 'myocyte_map_from_old_param_name_to_new_name_dict.txt')
myo_path_to_html = os.path.join(project_path, 'MyocyteSystem')
latex_eq_path = os.path.join(problem_folder,'eqs.txt')
sysv1_path_to_solution = os.path.join(project_path, 'SystemV1')

adipocyte_translators_of_names_path = os.path.join(problem_folder, 'adipocyte_translators')
adipocyte_map_from_old_param_name_to_new_name_dict_filename = os.path.join(myocyte_translators_of_names_path, 'adipocyte_map_from_old_param_name_to_new_name_dict.txt')
adipocyte_map_from_old_y_name_to_new_name_dict_filename = os.path.join(myocyte_translators_of_names_path, 'adipocyte_map_from_old_y_name_to_new_name_dict.txt')

