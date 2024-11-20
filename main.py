import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

from load_data import SuntansData, interpolate_field
from process_data import *

matplotlib.use('qtagg')
gridspec
interpolate_field

# rundata_folder = '/home/pkuznetsov/lmnad/data-success/'
# rundata_folder = '/home/pkuznetsov/lmnad/sec2/data/'
rundata_folder = '/home/pkuznetsov/lmnad/sec2_2/data/'
h = 0


def plt_set_fonts():
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('text', usetex=False)
    plt.rc('text.latex', preamble=r'\renewcommand{\rmdefault}{ftm}'
                                  r'\usepackage[T1,T2A]{fontenc}'
                                  r'\usepackage[utf8]{inputenc}'
                                  r'\usepackage[english,main=russian]{babel}'
                                  r'\usepackage{newtxmath,newtxtext}')


cb = None


def plot_all_for_step(step, ax0, ax1, ax2, ax3, draw_cb):
    global cb
    s, t, u, v, w = dl.load_data_for_step(step)

    ax0.cla()
    ax1.cla()
    ax2.cla()
    ax3.cla()
    if cb is not None:
        cb.remove()

    c = ax0.pcolormesh(dl.x, dl.z, s, cmap='jet')
    ax0.invert_yaxis()
    ax0.set_title("s(x,z,t), t={:.2f} ч".format(dl.step_to_h(step)))
    ax0.set_ylabel("Глубина, м")
    if draw_cb:
        cb1 = plt.figure(1).colorbar(c, ax=ax0)
        cb1.ax.set_title("аном. плот.")

    # c = ax3.contourf(dl.x, dl.z, s, cmap='jet', levels=60)
    # ax3.set_xlabel("Расстояние, км")
    # ax3.set_ylabel("Глубина, м")
    # ax3.set_xlim([55,75])
    # ax3.set_ylim([0,175])
    # ax3.invert_yaxis()
    # fig.colorbar(c, ax=ax3)
    diff = calculate_diff_by_z(u, dl.z)
    c = ax3.contourf(dl.x, dl.z, diff, cmap='jet', levels=60)
    ax3.invert_yaxis()
    ax3.set_title("du/dz(x,z,t), t={:.2f} ч".format(dl.step_to_h(step)))
    ax3.set_xlabel("Расстояние, км")
    ax3.set_ylabel("Глубина, м")
    # c.set_clim(-0.02, 0.02)
    # нужно перерисовывать из-за contourf вместо pcolormesh
    # pcolormesh сильно менее информативный
    # if draw_cb:
    cb = plt.figure(2).colorbar(c, ax=ax3)
    cb.ax.set_title("м/с^2")

    c = ax1.pcolormesh(dl.x, dl.z, u, cmap='jet')
    ax1.invert_yaxis()
    ax1.set_title("u(x,z,t), t={:.2f} ч".format(dl.step_to_h(step)))
    c.set_clim(-0.5, 0.5)
    if draw_cb:
        cb1 = plt.figure(3).colorbar(c, ax=ax1)
        cb1.ax.set_title("м/c")

    c = ax2.pcolormesh(dl.x, dl.z, w, cmap='jet')
    ax2.invert_yaxis()
    ax2.set_title("w(x,z,t), t={:.2f} ч".format(dl.step_to_h(step)))
    ax2.set_xlabel("Расстояние, км")
    c.set_clim(-0.05, 0.05)
    if draw_cb:
        cb1 = plt.figure(4).colorbar(c, ax=ax2)
        cb1.ax.set_title("м/с")


plt_set_fonts()

dl = SuntansData(rundata_folder)
dl.load_common_values()
# s, t, u, v, w = dl.load_data_for_step(0)
#
# abs_u_max = np.zeros(u.shape, dtype=np.float64)
# abs_w_max = np.zeros(w.shape, dtype=np.float64)
# abs_m_max = np.zeros(w.shape, dtype=np.float64)
#
# abs_u_max[np.isnan(u)] = np.NaN
# abs_w_max[np.isnan(w)] = np.NaN
# abs_m_max[np.isnan(w) | np.isnan(u)] = np.NaN
#
# for step in np.arange(dl.h_to_step(24 * 2)):
#     s, t, u, v, w = dl.load_data_for_step(step)
#
#     abs_greater = np.abs(abs_u_max) < np.abs(u)
#     abs_u_max[abs_greater] = u[abs_greater]
#     abs_greater = np.abs(abs_w_max) < np.abs(w)
#     abs_w_max[abs_greater] = w[abs_greater]
#     m = np.sqrt(u ** 2 + w ** 2)
#     abs_greater = abs_m_max < m
#     abs_m_max[abs_greater] = m[abs_greater]
#
# fig = plt.figure()
# fig.show()
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
#
# gs = gridspec.GridSpec(2, 2, hspace=0.2, wspace=0.2)
# ax0 = fig.add_subplot(gs[0, 0])
# ax1 = fig.add_subplot(gs[1, 0])
# ax2 = fig.add_subplot(gs[:, 1])
#
# x_cut = dl.x[0:len(dl.x) - 10]
#
# c = ax0.pcolormesh(x_cut, dl.z, np.abs(abs_u_max[:, 0:len(dl.x) - 10]), cmap='jet')
# ax0.set_title(f"Максимальная гор. скорость ")
# ax0.set_ylabel("Глубина, м")
# ax0.invert_yaxis()
# plt.colorbar(c, ax=ax0)
#
# c = ax1.pcolormesh(x_cut, dl.z, np.abs(abs_w_max[:, 0:len(dl.x) - 10]), cmap='jet')
# ax1.set_title(f"Максимальная верт. скорость")
# ax1.set_xlabel("Расстояние, км")
# ax1.set_ylabel("Глубина, м")
# ax1.invert_yaxis()
# # c.set_clim(0, 0.02)
# plt.colorbar(c, ax=ax1)
#
# c = ax2.pcolormesh(x_cut, dl.z, abs_m_max[:, 0:len(dl.x) - 10],
#                    cmap='jet')
# ax2.set_title(f"Максимальный модуль скорости")
# ax2.set_xlabel("Расстояние, км")
# ax2.set_ylabel("Глубина, м")
# ax2.invert_yaxis()
# # c.set_clim(0, 0.02)
# plt.colorbar(c, ax=ax2)
#
# print(
#     f"u_max = {np.nanmax(np.abs(abs_u_max[:, 0:len(dl.x) - 10]))}, w_max = {np.nanmax(np.abs(abs_w_max[:, 0:len(dl.x) - 10]))}")
#
# plt.gcf().set_size_inches(15, 8)
# plt.savefig(
#     f"speeds_u_max={np.nanmax(np.abs(abs_u_max[:, 0:len(dl.x) - 10])):.2f}_w_max={np.nanmax(np.abs(abs_w_max[:, 0:len(dl.x) - 10])):.2f}.png",
#     dpi=500, bbox_inches="tight")
#
# plt.show()

# print(dl.x)
# x = [70, 85]  # data-suc
# x = [65, 80]  # sec2
x = [100, 110]  # sec3

# u_x, s_x, steps = dl.get_u_s_for_x(x)
#
# fig = plt.figure()
# fig.show()
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
#
# gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)
# ax0 = fig.add_subplot(gs[0, 0])
# ax1 = fig.add_subplot(gs[1, 0])
# ax2 = fig.add_subplot(gs[0, 1])
# ax3 = fig.add_subplot(gs[1, 1])
#
# t_arr = np.linspace(0, dl.step_to_h(steps - 1), steps)
# c = ax0.pcolormesh(t_arr, dl.z, np.transpose(u_x[0]), cmap='jet')
# ax0.set_title(f"Гор. скорость в точке x={x[0]:.2f}")
# ax0.set_xlabel("Время, ч")
# ax0.set_ylabel("Глубина, м")
# ax0.invert_yaxis()
# plt.colorbar(c, ax=ax0)
#
# c = ax1.pcolormesh(t_arr, dl.z, np.transpose(u_x[1]), cmap='jet')
# ax1.set_title(f"Гор. скорость в точке x={x[1]:.2f}")
# ax1.set_xlabel("Время, ч")
# ax1.set_ylabel("Глубина, м")
# ax1.invert_yaxis()
# plt.colorbar(c, ax=ax1)
#
# c = ax2.pcolormesh(t_arr, dl.z, np.transpose(s_x[0]), cmap='jet')
# ax2.set_title(f"Плотность в точке x={x[0]:.2f}")
# ax2.set_xlabel("Время, ч")
# ax2.set_ylabel("Глубина, м")
# ax2.invert_yaxis()
# plt.colorbar(c, ax=ax2)
#
# c = ax3.pcolormesh(t_arr, dl.z, np.transpose(s_x[1]), cmap='jet')
# ax3.set_title(f"Плотность в точке x={x[1]:.2f}")
# ax3.set_xlabel("Время, ч")
# ax3.set_ylabel("Глубина, м")
# ax3.invert_yaxis()
# plt.colorbar(c, ax=ax3)
#
# F_n_old, M_old, steps, fn_d_old, fn_i_old = dl.calc_loads_momentum(x, 1025.09, 0, np.Inf, False)
# t_arr = np.linspace(0, dl.step_to_h(steps - 1), steps)
# t_cut = 10
# F_n_old[:, 0:t_cut] = np.nan
# M_old[:, 0:t_cut] = np.nan
# # fn_d[:, 0:t_cut, :] = np.nan
# # fn_i[:, 0:t_cut, :] = np.nan
#
# F_n_new, M_new, steps, fn_d_new, fn_i_new = dl.calc_loads_momentum(x, 1025.09, 0, np.Inf, True)
# F_n_new[:, 0:t_cut] = np.nan
# M_new[:, 0:t_cut] = np.nan
#
# fig = plt.figure()
# fig.show()
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.15)
# ax0 = fig.add_subplot(gs[0, 0])
# ax1 = fig.add_subplot(gs[0, 1])
# ax2 = fig.add_subplot(gs[1, 0])
# ax3 = fig.add_subplot(gs[1, 1])
#
# ax0.plot(t_arr / 12.42, F_n_old[0] / 100000, color='blue', label='(1)')
# ax2.plot(t_arr / 12.42, M_old[0] / 1000000, color='blue', label='(1)')
# ax1.plot(t_arr / 12.42, F_n_old[1] / 100000, color='blue', label='(1)')
# ax3.plot(t_arr / 12.42, M_old[1] / 1000000, color='blue', label='(1)')
#
# ax0.plot(t_arr / 12.42, F_n_new[0] / 100000, color='red', label='(2)')
# ax2.plot(t_arr / 12.42, M_new[0] / 1000000, color='red', label='(2)')
# ax1.plot(t_arr / 12.42, F_n_new[1] / 100000, color='red', label='(2)')
# ax3.plot(t_arr / 12.42, M_new[1] / 1000000, color='red', label='(2)')
#
# ax0.set_title(f"В точке x={x[0]:.2f}")
# ax0.set_xlim([0.3, 5.5])
# ax0.set_ylabel("F_n(t), 10^5*Н")
# ax0.legend()
# ax0.grid()
#
# ax1.set_title(f"В точке x={x[1]:.2f}")
# ax1.set_xlim([0.3, 5.5])
# ax1.legend()
# ax1.grid()
#
# ax2.set_xlim([0.3, 5.5])
# ax2.legend()
# ax2.set_ylabel("M(t), MН*м ")
# ax2.set_xlabel("Время/T_M2")
# ax2.grid()
#
# ax3.set_xlim([0.3, 5.5])
# ax3.set_xlabel("Время/T_M2")
# ax3.legend()
# ax3.grid()
#
# plt.gcf().set_size_inches(15, 8)
# plt.savefig(
#     f"loads_1.png",
#     dpi=500, bbox_inches="tight")
#
# ind_0 = np.nanargmax(np.abs(F_n_old[0]))
# ind_1 = np.nanargmax(np.abs(F_n_old[1]))
#
# t0 = 4.85
# t1 = 4.75
# ind_0 = np.nanargmin(np.abs(t_arr - t0 * 12.42))
# ind_1 = np.nanargmin(np.abs(t_arr - t1 * 12.42))
#
# fig = plt.figure()
# fig.show()
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.15)
# ax0 = fig.add_subplot(gs[0, 0])
# ax1 = fig.add_subplot(gs[0, 1])
# ax2 = fig.add_subplot(gs[1, 0])
# ax3 = fig.add_subplot(gs[1, 1])
#
# ax0.barh(dl.z, fn_d_old[0, ind_0, :], color='blue', label='(1)', height=2.5, alpha=0.5)
# ax2.barh(dl.z, fn_i_old[0, ind_0, :], color='blue', label='(1)', height=2.5, alpha=0.5)
# ax1.barh(dl.z, fn_d_old[1, ind_1, :], color='blue', label='(1)', height=2.5, alpha=0.5)
# ax3.barh(dl.z, fn_i_old[1, ind_1, :], color='blue', label='(1)', height=2.5, alpha=0.5)
#
# ax0.barh(dl.z, fn_d_new[0, ind_0, :], color='red', label='(2)', height=2.5, alpha=0.5)
# ax2.barh(dl.z, fn_i_new[0, ind_0, :], color='red', label='(2)', height=2.5, alpha=0.5)
# ax1.barh(dl.z, fn_d_new[1, ind_1, :], color='red', label='(2)', height=2.5, alpha=0.5)
# ax3.barh(dl.z, fn_i_new[1, ind_1, :], color='red', label='(2)', height=2.5, alpha=0.5)
#
# ax0.set_title(f"В точке x={x[0]:.2f}, t={t0} 1/T_M2")
# # ax0.set_title(f"В точке x={x[0]:.2f}")
# ax0.set_ylabel("Глубина, м\nF_D(z)")
# ax0.invert_yaxis()
# ax0.legend()
# ax0.grid()
#
# ax1.set_title(f"В точке x={x[1]:.2f}, t={t1} 1/T_M2")
# # ax1.set_title(f"В точке x={x[1]:.2f}")
# ax1.invert_yaxis()
# ax1.legend()
# ax1.grid()
#
# ax2.set_ylabel("Глубина, м\nF_I(z)")
# ax2.invert_yaxis()
# ax2.legend()
# ax2.set_xlabel("Н/м")
# ax2.grid()
#
# ax3.set_xlabel("Н/м")
# ax3.legend()
# ax3.grid()
#
# plt.gcf().set_size_inches(15, 8)
# plt.savefig(
#     f"loads_2.png",
#     dpi=500, bbox_inches="tight")

# ax0.barh(dl.z, fn_d[0, ind, :], color='yellow', label='F_D(z), Н/м')
# ax0.set_ylabel("Глубина, м")
# ax0.invert_yaxis()
# ax0.barh(dl.z, fn_i[0, ind, :], color='blue', label='F_I(z), Н/м')
# ax0.set_title(f"В точке x={x[0]:.2f}")
# ax0.legend()
# ax0.grid()
#
# ind = np.nanargmax(np.abs(F_n[1]))
# ax1.barh(dl.z, fn_d[1, ind, :], color='yellow', label='F_D(z), Н/м')
# ax1.invert_yaxis()
# ax1.set_title(f"В точке x={x[1]:.2f}")
# ax1.barh(dl.z, fn_i[1, ind, :], color='blue', label='F_I(z), Н/м')
# ax1.legend()
# ax1.grid()
#
# plt.gcf().set_size_inches(15, 8)
# plt.savefig(
#     f"loads_2.png",
#     dpi=500, bbox_inches="tight")

# plt.show()
# exit(0)
#
# fig = plt.figure()
# fig.show()
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
#
# gs = gridspec.GridSpec(2, 2, hspace=0.2, wspace=0.2)
# ax0 = fig.add_subplot(gs[0, 0])  # row 0, col 0
# ax1 = fig.add_subplot(gs[0, 1])  # row 1, col 0
# ax2 = fig.add_subplot(gs[1, 1])  # row 1, col 1
# ax3 = fig.add_subplot(gs[1, 0])  # row 0, col 1
plt.figure(1)
ax0 = plt.gca()
plt.figure(2)
ax1 = plt.gca()
plt.figure(3)
ax2 = plt.gca()
plt.figure(4)
ax3 = plt.gca()

step = dl.h_to_step(h)
plot_all_for_step(step, ax0, ax1, ax2, ax3, True)
plt.gcf().set_size_inches(15, 8)

fig = plt.figure()
max = int(dl.nsteps / dl.ntout)
data = np.zeros((max, len(dl.x)))
for step in range(0, max):
    fs = dl.load_h(step)
    data[step, :] = fs
c = plt.contourf(dl.x, dl.step_to_h(np.arange(0, max)), data, cmap="jet", levels=60)
cb1 = fig.colorbar(c)
cb1.ax.set_title("м")
plt.xlabel("Расстояние, км")
plt.ylabel("Время, ч")
plt.show()

fig = plt.figure()
plt.ion()
for step in range(max):
    fs = dl.load_h(step)
    plt.plot(dl.x, fs)
    plt.ylim([-20, 20])
    plt.pause(0.15)
    if step < max - 1:
        fig.clf()
# # plt.show()
# plt.savefig(f"out_{h:.2f}.png", dpi=500, bbox_inches="tight")
#
# plt.figure()
# s, t, u, v, w = dl.load_data_for_step(step)
# # c = plt.gca().pcolormesh(dl.x, dl.z, s, cmap='jet')
# c = plt.gca().contourf(dl.x, dl.z, s, cmap='jet', levels=75)
# plt.gca().set_title("s(x,z,t), t={:.2f} ч".format(dl.step_to_h(step)))
# plt.gca().set_ylabel("Глубина, м")
# plt.xlim((19, 61))
# plt.ylim((0, 600))
# plt.gca().invert_yaxis()
# cb1 = fig.colorbar(c, ax=plt.gca())
# cb1.ax.set_title("аном. плот.")
# plt.gcf().set_size_inches(15, 8)
# # plt.show()
# plt.savefig(f"out_det_{h:.2f}.png", dpi=500, bbox_inches="tight")
plt.ioff()
plt.show()

do_tight = True
draw_cb = True
os.mkdir("video")
# plt.draw()

plt.figure(1)
ax0 = plt.gca()
plt.figure(2)
ax1 = plt.gca()
plt.figure(3)
ax2 = plt.gca()
plt.figure(4)
ax3 = plt.gca()

for s in np.arange(dl.sec_to_step(dl.runtime_sec())):
    plot_all_for_step(s, ax0, ax1, ax2, ax3, draw_cb)
    # if do_tight:
    #     gs.tight_layout(fig)
    plt.pause(0.001)
    # plt.draw()
    plt.figure(1)
    plt.savefig(f"video/s{s:03d}.png", dpi=750)
    plt.figure(1)
    plt.savefig(f"video/u{s:03d}.png", dpi=750)
    plt.figure(1)
    plt.savefig(f"video/w{s:03d}.png", dpi=750)
    plt.figure(1)
    plt.savefig(f"video/du{s:03d}.png", dpi=750)
    draw_cb = False
    # if s == 3:
    #     do_tight = False

quit()
# count = 0
# line = np.zeros(dl.x.size, dtype=np.float64)
# for step in range(2, 35):
#     s, t, u, v, w = dl.load_data_for_step(step)
#     diff = calculate_diff_by_z(u, dl.z)
#     print(f"step: {step}")
#     for ix in np.arange(dl.x.size):
#         maxdu = 0
#         counter = 0
#         max_i = 0
#         diff_col = diff[:, ix]
#         depth = 0
#         for iz in np.flip(np.arange(dl.z.size)):
#             if ~np.isnan(s[iz, ix]) and depth == 0:
#                 depth = dl.z[iz]
#             if ~np.isnan(diff_col[iz]):
#                 counter += 1
#                 if np.abs(diff_col[iz]) >= maxdu:
#                     max_i = iz
#                     # counter = 0
#                     maxdu = np.abs(diff_col[iz])
#                 else:
#                     if counter > 10:
#                         break
#         line[ix] += (depth - dl.z[max_i])
#     count += 1
#
# for ix in np.arange(dl.x.size):
#     line[ix] /= count
#
# plt.plot(dl.x, savgol_filter(line, 35, 3))
# plt.xlabel("Расстояние, км")
# plt.ylabel("Толщина, м")
# plt.title("Толщина пограничного слоя")
# plt.xlim(0, dl.x[-1])
# plt.tight_layout()
# ax0.set_ylim(0, 100)


#     plt.gcf().clear()
#
#     gs = gridspec.GridSpec(2, 1, hspace=0.2, wspace=0.2)
#     ax0 = fig.add_subplot(gs[0, 0])
#     ax1 = fig.add_subplot(gs[1, 0])
#
#     ax0.plot(dl.x, line)
#     ax0.set_xlabel("Расстояние, км")
#     ax0.set_ylabel("Толщина, м")
#     ax0.set_title("Толщина пограничного слоя")
#     ax0.set_xlim(0, dl.x[-1])
#     # ax0.set_ylim(0, 100)
#
#     c = ax1.contourf(dl.x, dl.z, np.abs(diff), cmap='jet', levels=60)
#     ax1.invert_yaxis()
#     ax1.set_title("du/dz(x,z,t), t={:.2f} ч".format(dl.step_to_h(step)))
#     ax1.set_xlabel("Расстояние, км")
#     ax1.set_ylabel("Глубина, м")
#     # cb = fig.colorbar(c, ax=ax1)
#
#     gs.tight_layout(fig)
#
#     plt.pause(0.01)
# #
# plt.figure()
# U = 0
# TIDE_TIME_SHIFT = 0
# time = np.linspace(0, 30 * 24 * 60 * 60, 1000)
# M_PI = np.pi
# Asin_m2 = 1.133329e-02
# Asin_s2 = -4.847622e-04
# Asin_k1 = -2.112834e-02
# Asin_o1 = -1.130596e-02
# Asin_p1 = -6.584461e-03
# Asin_q1 = -1.558616e-03
#
# Acos_m2 = 9.246401e-03
# Acos_s2 = 4.650254e-03
# Acos_k1 = 5.675759e-04
# Acos_o1 = 1.100382e-02
# Acos_p1 = 9.158928e-04
# Acos_q1 = 3.010823e-03
#
# U += Asin_m2 * sin((2 * M_PI) / (12.42 * 3600) * (time + TIDE_TIME_SHIFT)) + Acos_m2 * cos(
#     (2 * M_PI) / (12.42 * 3600) * (time + TIDE_TIME_SHIFT))
# U += Asin_s2 * sin((2 * M_PI) / (12.00 * 3600) * (time + TIDE_TIME_SHIFT)) + Acos_s2 * cos(
#     (2 * M_PI) / (12.00 * 3600) * (time + TIDE_TIME_SHIFT))
# U += Asin_k1 * sin((2 * M_PI) / (23.93 * 3600) * (time + TIDE_TIME_SHIFT)) + Acos_k1 * cos(
#     (2 * M_PI) / (23.93 * 3600) * (time + TIDE_TIME_SHIFT))
# U += Asin_p1 * sin((2 * M_PI) / (24.07 * 3600) * (time + TIDE_TIME_SHIFT)) + Acos_p1 * cos(
#     (2 * M_PI) / (24.07 * 3600) * (time + TIDE_TIME_SHIFT))
# U += Asin_o1 * sin((2 * M_PI) / (25.82 * 3600) * (time + TIDE_TIME_SHIFT)) + Acos_o1 * cos(
#     (2 * M_PI) / (25.82 * 3600) * (time + TIDE_TIME_SHIFT))
# U += Asin_q1 * sin((2 * M_PI) / (26.87 * 3600) * (time + TIDE_TIME_SHIFT)) + Acos_q1 * cos(
#     (2 * M_PI) / (26.87 * 3600) * (time + TIDE_TIME_SHIFT))
# plt.plot(time / (60 * 60 * 24), U)
# plt.xlabel("t, дни")
# plt.ylabel("U, м/с")
#
# plt.gcf().set_size_inches(15, 8)
# plt.show()

# iso = get_isocline(dl.x, dl.z, s, 0.0267)
# plt.figure()
# plt.plot(dl.x, dl.profile)
# plt.plot(dl.x, iso)
# plt.gca().invert_yaxis()
#
x_iso = dl.get_x_slice(0, dl.x[len(dl.x) - 10])
# x_iso = dl.x
iso, steps = dl.calculate_isoclines(x_iso, 60)
np.savetxt(rundata_folder + 'iso.txt', iso)
# print(steps)
# iso = np.loadtxt('iso.txt')
# steps = 801
iso_interp_factor = 2
iso, t_arr, x_arr = interpolate_field(np.linspace(0, dl.step_to_h(steps - 1), steps), x_iso, iso,
                                      x_factor=iso_interp_factor, y_factor=iso_interp_factor)
plt.figure()
c = plt.pcolormesh(x_arr, t_arr, -iso, cmap='jet')
plt.title("Изопикны")
plt.xlabel("Расстояние, км")
plt.ylabel("Время, ч")
# c.set_clim(-20, 20)
plt.colorbar(c, ax=plt.gca())

plt.gcf().set_size_inches(15, 8)
plt.savefig(
    f"iso.png",
    dpi=500, bbox_inches="tight")
plt.show()

# plt.figure()
# plot = iso[:, 50]
# plt.plot(range(0, iso.shape[0]), plot)
# # plot = prepare_fft_sample(plot, 1 / dl.step_to_sec(1) * iso_interp_factor)
# plt.plot(range(0, iso.shape[0]), plot)
# plt.title("Изопикна")
# plt.xlabel("Номер отсчета по времени")
# plt.ylabel("Смещение, м")
# plt.figure()
# t_points_count = len(plot)
# if t_points_count % 2 == 0:
#     fft_size = round(t_points_count / 2) + 1
# else:
#     fft_size = round((t_points_count + 1) / 2)
# fft = np.abs(np.fft.rfft(plot)) / fft_size
# f = np.zeros(fft_size, dtype=np.float64)
# dt = dl.dt / iso_interp_factor
# f[:] = 2 * np.pi / dt * np.arange(fft_size) / t_points_count
# plt.plot(f, fft)
# plt.xscale('log')
# plt.ylabel(f"Частота, Гц")
# plt.show()

dx_kmeters = (dl.x[1] - dl.x[0])
spectr, k = calculate_spectrum_x(iso, dx_kmeters / iso_interp_factor,
                                 1 / dl.step_to_sec(1) * iso_interp_factor)
plt.figure()
c = plt.pcolormesh(k, t_arr, spectr, cmap='jet')
plt.title("Спектр по оси x")
fs = 1 / dx_kmeters
plt.xlabel(f"Волновое число, 1/км (Частота дискретизации {fs:.2f})")
plt.ylabel("Время, ч")
# c.set_clim(-0.05, 0.05)
# до частоты дискретизации/2 по x
# plt.xlim(0, 1 / dx_kmeters / 2)
plt.xlim(0, 0.5)
# plt.xscale('log')
plt.colorbar(c, ax=plt.gca())

spectr, f = calculate_spectrum_t(iso, dl.dt / iso_interp_factor, 1 / (dx_kmeters * 1000) * iso_interp_factor)
plt.figure()
c = plt.pcolormesh(x_arr, f, spectr, cmap='jet')
plt.title("Спектр по времени")
fs = 1 / dl.dt
plt.ylabel(f"Частота, Гц (Частота дискретизации {fs:.2f})")
plt.xlabel("Расстояние, км")
# c.set_clim(-0.05, 0.05)
# plt.ylim(0, fs / 2)
plt.ylim(0, 0.1)
# plt.yscale('log')
plt.colorbar(c, ax=plt.gca())

plt.show()
