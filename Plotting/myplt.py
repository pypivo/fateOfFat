import os.path

import numpy as np
import plotly
import plotly.graph_objects as go

from ParsingSystem.parse_and_build import make_params_vec_from_params_dict, get_start_point_values, \
    get_y_name_by_index_of_solution

import matplotlib.pyplot as plt


def plot_vec(x, y,title='', block=True):
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    ax.set_title(title)
    if block == True:
        fig.waitforbuttonpress(timeout=-1)
    else:
        fig.show()


def init_figure(x_label='$t,s$',y_label=r'$n, \frac{mol}{L}$'):
    plotly.io.templates.default = 'plotly_dark'
    fig = go.Figure(
            layout=go.Layout(
            # title="Mt Bruno Elevation",
            xaxis_title=x_label,
            yaxis_title=y_label,
            # xaxis=dict(rangeslider=dict(visible=True)), # add slider
            # width=1900, height=4000
            # margin=dict(
            #     l=0,
            #     r=0,
            #     b=0,
            #     t=0,
            #     pad=4
            # ),
        ))
    return fig
def plot_solutions(fig, solutions, time_grid, name_by_index):

    # https://towardsdatascience.com/8-visualizations-with-python-to-handle-multiple-time-series-data-19b5b2e66dd0

    # plotly (javascript)
    # 'plotly_dark','none'

    number_of_solutions = solutions.shape[1]
    for i_of_sol in range(number_of_solutions):
        line_name= name_by_index[i_of_sol]
        ith_solution = solutions[:, i_of_sol]
        line_style = None
        if np.sum(np.where(ith_solution<0))>0:
            line_style ='dot'

        number_of_points = ith_solution.shape[0]
        target_number_of_points = min(1000,number_of_points)
        step_ = int(number_of_points/target_number_of_points)
        fig.add_trace(go.Scatter(x=time_grid[::step_],
                                 y=ith_solution[::step_],
                                 name='$'+line_name+'$',
                                 fill=None,
                                 line=dict(width=4, dash=line_style)
                                 )
                      )
    # fig.update_layout(xaxis_title='$t,s$',
    #                   yaxis_title='$n$')
    # save plotly figure to html
    # fig.write_html('tmp.html')
    # open figure in browser
    return fig

    # plotly multiple plots in one html
    # with open(config.plotly_plotting_html, 'w') as f:
    #     number_of_solutions = solutions.shape[1]
    #     for i_of_sol in range(number_of_solutions):
    #
    #
    #
    #         line_name = get_y_name_by_index_of_solution(index_of_solution= i_of_sol,
    #                                                     from_new_to_old_y_names_dict=names_dict)
    #
    #
    #         ith_solution = solutions[:, i_of_sol]
    #         line_style = None
    #         if np.sum(np.where(ith_solution<0))>0:
    #             line_style ='dot'
    #
    #         fig_i = go.Figure(
    #             layout=go.Layout(
    #                 # title="Mt Bruno Elevation",
    #                 xaxis_title='$t,s$',
    #                 yaxis_title=r'$n, \frac{mol}{L}$',
    #                 title='$'+line_name+'$',
    #                 # xaxis=dict(rangeslider=dict(visible=True)), # add slider
    #                 # width=1900, height=4000
    #                 # margin=dict(
    #                 #     l=0,
    #                 #     r=0,
    #                 #     b=0,
    #                 #     t=0,
    #                 #     pad=4
    #                 # ),
    #             ))
    #
    #         fig_i.add_trace(go.Scatter(x=time_grid,
    #                                  y=ith_solution,
    #                                  name='$'+line_name+'$',
    #                                  fill=None,
    #                                  line=dict(width=4, dash=line_style)
    #                                  )
    #                       )
    #
    #         f.write(fig_i.to_html(full_html=False, include_plotlyjs='cdn'))


    # matplotlib
    # plt.rcParams["figure.figsize"] = [14, 7]
    # plt.rcParams["figure.autolayout"] = True
    # fig = plt.figure()
    # axs = fig.add_subplot(111)
    # number_of_solutions = solutions.shape[1]
    # for i_of_sol in range(number_of_solutions):
    #
    #     line_name = get_y_name_by_index_of_solution(index_of_solution= i_of_sol,
    #                                                 from_new_to_old_y_names_dict=names_dict)
    #     ith_solution = solutions[:, i_of_sol]
    #     if np.sum(np.where(ith_solution<0))>0:
    #         axs.plot(time_grid, ith_solution, label = '$'+line_name+'$', linestyle='dashed')
    #     else:
    #         axs.plot(time_grid, ith_solution, label = '$'+line_name+'$')
    # axs.legend(bbox_to_anchor=(1.04, 1), loc="upper left",
    #            ncol=2,
    #            fancybox=True, shadow=True
    #            )
    # axs.set_xlabel(r'$t$')
    # axs.set_ylabel(r'$n$')
    # axs.grid()
    # plt.show()

def add_line_to_fig(fig, time_grid, y: np.array, line_name, line_style=None):
    number_of_points = y.shape[0]
    target_number_of_points = min(1000, number_of_points)
    step_ = int(number_of_points / target_number_of_points)
    fig.add_trace(go.Scatter(x=time_grid[::step_],
                             y=y[::step_],
                             name='$' + line_name + '$',
                             fill=None,
                             line=dict(width=4, dash=line_style)
                             )
                  )

def save_fig_to_html(fig, path, filename):

    if not os.path.exists(path):
        os.makedirs(path)

    fig.write_html(os.path.join(path,filename))

def add_scroll_bar(fig):
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True),
                                 type="linear"))