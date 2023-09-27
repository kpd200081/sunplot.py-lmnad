import numpy as np
from scipy.interpolate import interp1d, interpn

from process_data import get_isocline
from sfoda_utils import Grid


def index_arr_by2axis(arr, ind1, ind2):
    return arr.T[ind1][ind2].T


def interpolate_by2points(arr, indsort, ind1, ind2):
    return (index_arr_by2axis(arr, indsort, ind1) + index_arr_by2axis(arr, indsort, ind2)) / 2


class SuntansData(object):
    sungrid = None
    dat_file = {}
    dt = 0
    ntout = 0
    nsteps = 0
    rundata_folder = ""
    sizeofdouble = 8
    EMPTY = 999999
    depth = None
    s0 = None

    sortx = None
    y1 = None
    y2 = None
    y1inds = None
    y2inds = None

    x = None
    z = None
    profile = None

    def __init__(self, path, suntans_dat="suntans.dat"):
        self.rundata_folder = path
        self.dat_file = self.load_sundat_file(suntans_dat)

    def load_common_values(self):
        self.dt = float(self.dat_file["dt"])
        self.ntout = int(self.dat_file["ntout"])
        self.nsteps = int(self.dat_file["nsteps"])
        self.sungrid = Grid(self.rundata_folder, VERBOSE=True, Nk=0)
        self.sungrid.loadBathy(self.rundata_folder + "depth.dat-voro")

        self.depth = np.multiply(range(0, self.sungrid.Nkmax), self.sungrid.dz.T).T
        self.s0 = np.fromfile(self.rundata_folder + "s0.dat", dtype=np.double)
        self.s0 = self.s0.reshape(self.sungrid.Nkmax, self.sungrid.Nc)
        self.s0[self.s0 == self.EMPTY] = np.NaN

        self.sortx = np.argsort(self.sungrid.xv)
        self.y1 = np.min(self.sungrid.yv)
        self.y2 = np.max(self.sungrid.yv)
        self.y1inds = self.sungrid.yv[self.sortx] == self.y1
        self.y2inds = self.sungrid.yv[self.sortx] == self.y2

        self.s0 = interpolate_by2points(self.s0, self.sortx, self.y1inds, self.y2inds)

        self.x = self.sungrid.xv[self.sortx][self.y1inds] / 1000
        self.z = self.sungrid.z_r
        self.profile = self.sungrid.dv[self.sortx][self.y1inds]

    def step_to_sec(self, step):
        return step * self.dt * self.ntout

    def sec_to_step(self, sec):
        return round(sec / (self.dt * self.ntout))

    def step_to_h(self, step):
        return self.step_to_sec(step) / 3600

    def h_to_step(self, h):
        return self.sec_to_step(h * 3600)

    def runtime_sec(self):
        return round(self.nsteps * self.dt)

    def load_data_for_step(self, step):
        s = self.load_data_file("s", step)
        T = self.load_data_file("T", step)
        U = self.load_data_file("u", step, 3)
        u = U[0, :, :]
        v = U[1, :, :]
        w = U[2, :, :]
        s = interpolate_by2points(s, self.sortx, self.y1inds, self.y2inds)
        T = interpolate_by2points(T, self.sortx, self.y1inds, self.y2inds)
        u = interpolate_by2points(u, self.sortx, self.y1inds, self.y2inds)
        v = interpolate_by2points(v, self.sortx, self.y1inds, self.y2inds)
        w = interpolate_by2points(w, self.sortx, self.y1inds, self.y2inds)
        return s, T, u, v, w

    def load_sundat_file(self, file):
        ret = {}
        f = self.rundata_folder + file
        f = open(f, 'r')
        lines = f.readlines()
        for line in lines:
            line = line.replace("\t", " ")
            strip = line.partition("#")[0]
            if strip != '':
                part = strip.partition(" ")
                key = part[0]
                val = part[2].replace(" ", "")
                ret[key] = val
        return ret

    def load_data_file(self, file, step, components=1):
        f = self.rundata_folder + file + ".dat"
        common_seek = step * self.sungrid.Nkmax * self.sungrid.Nc * self.sizeofdouble * components
        if components > 1:
            arr = np.fromfile(f, dtype=np.double,
                              count=self.sungrid.Nkmax * self.sungrid.Nc * components,
                              offset=common_seek)
            arr = arr.reshape(components, self.sungrid.Nkmax, self.sungrid.Nc)
        else:
            arr = np.fromfile(f, dtype=np.double,
                              count=self.sungrid.Nkmax * self.sungrid.Nc,
                              offset=common_seek)
            arr = arr.reshape(self.sungrid.Nkmax, self.sungrid.Nc)
        arr[arr == self.EMPTY] = np.NaN
        return arr

    def get_x_slice(self, x_from, x_to):
        return self.x[np.argmin(np.abs(self.x - x_from)):np.argmin(np.abs(self.x - x_to))]

    def calculate_isoclines(self, x_array, h0, t_from=0, t_to=np.Inf, depth_int=50):
        steps = 0
        if t_to == np.Inf:
            t_to = self.runtime_sec()
        step_start = self.sec_to_step(t_from)
        step_end = self.sec_to_step(t_to)
        # f = interp1d(range(0, len(self.z)), self.z, kind='slinear')
        # z_array = f(np.linspace(0, len(self.z)-1, len(self.z)*50))
        # Z, X = np.meshgrid(z_array, self.x)
        # s_interp = interpn((self.z, self.x), self.s0, (Z, X), method='linear')
        # s_interp = np.transpose(s_interp)
        # plt.figure()
        # plt.plot(range(0, len(self.z)), self.z+1)
        # plt.plot(np.linspace(0, len(self.z)-1, len(self.z)*100), z_array)
        # plt.pcolormesh(X, Z, s_interp, shading='auto', cmap='jet')
        # plt.show()
        s_interp, z_array, _ = interpolate_field(self.z, self.x, self.s0, x_factor=depth_int)
        s0 = s_interp[np.argmin(np.abs(z_array - h0)), :]
        h0 = get_isocline(x_array, z_array, s_interp, s0)
        ret = np.zeros((step_end - step_start + 1, len(x_array)), dtype=np.float64)
        for t in range(step_start, step_end + 1):
            if steps % 50 == 0:
                print(f"Calculate isoclines, now at {100 * (t - step_start) / (step_end - step_start):.2f}%")
            s, _, _, _, _ = self.load_data_for_step(t)
            s_interp, _, _ = interpolate_field(self.z, self.x, s, x_factor=depth_int)
            h = get_isocline(x_array, z_array, s_interp, s0)
            ret[t - step_start, :] = h - h0
            steps += 1
        return ret, steps

    def get_u_w_s_for_x(self, x, t_from=0, t_to=np.Inf):
        steps = 0
        if type(x) is int or type(x) is float or type(x) is np.float64:
            x_inds = np.zeros(1, dtype=int)
            x_inds[0] = np.nanargmin(np.abs(self.x - x))
        else:
            x_inds = np.zeros(len(x), dtype=int)
            for i in range(0, len(x)):
                x_inds[i] = np.array(np.nanargmin(np.abs(self.x - x[i])), dtype=int)
        if t_to == np.Inf:
            t_to = self.runtime_sec()
        step_start = self.sec_to_step(t_from)
        step_end = self.sec_to_step(t_to)
        u_x = np.zeros((x_inds.size, step_end - step_start + 1, len(self.z)), dtype=np.float64)
        w_x = np.zeros((x_inds.size, step_end - step_start + 1, len(self.z)), dtype=np.float64)
        s_x = np.zeros((x_inds.size, step_end - step_start + 1, len(self.z)), dtype=np.float64)
        for t in range(step_start, step_end + 1):
            s, _, u, _, w = self.load_data_for_step(t)
            for i in range(0, x_inds.size):
                u_x[i, t - step_start, :] = u[:, x_inds[i]]
                w_x[i, t - step_start, :] = w[:, x_inds[i]]
                s_x[i, t - step_start, :] = s[:, x_inds[i]]
            steps += 1
        return u_x, w_x, s_x, steps

    def calc_loads_momentum(self, x, rho0, t_from=0, t_to=np.Inf, new_formula=True, with_sec_part=True):
        u_x, w_x, s_x, steps = self.get_u_w_s_for_x(x, t_from, t_to)
        rho = rho0 * (1 + s_x)
        Cd = 1
        Cm = 1
        R = 0.7
        g = 9.8
        if type(x) is int or type(x) is float:
            x_inds = np.zeros(1, dtype=int)
            x_inds[0] = np.nanargmin(np.abs(self.x - x))
        else:
            x_inds = np.zeros(len(x), dtype=int)
            for i in range(0, len(x)):
                x_inds[i] = np.array(np.nanargmin(np.abs(self.x - x[i])), dtype=int)
        F_n = np.zeros((x_inds.size, steps), dtype=np.float64)
        M = np.zeros((x_inds.size, steps), dtype=np.float64)
        fn_d = np.empty((x_inds.size, u_x.shape[1], self.z.size), dtype=np.float64)
        fn_i = np.empty((x_inds.size, u_x.shape[1], self.z.size), dtype=np.float64)
        fn_d.fill(np.nan)
        fn_i.fill(np.nan)
        for i in range(0, x_inds.size):
            nan_mask = ~np.isnan(u_x[i])
            real_u = u_x[i][nan_mask]
            real_rho = rho[i][nan_mask]
            z_ind = np.argmin(nan_mask[0, :])
            real_u = real_u.reshape(u_x.shape[1], z_ind)
            real_rho = real_rho.reshape(u_x.shape[1], z_ind)
            u_x_next, _, _, _ = self.get_u_w_s_for_x(self.x[x_inds[i] + 1], t_from, t_to)
            nan_mask_next = ~np.isnan(u_x_next)
            z_ind_next = np.argmin(nan_mask_next[0, :])
            real_u_next = u_x_next[nan_mask_next]
            real_u_next = real_u_next.reshape(u_x_next.shape[1], z_ind_next)
            real_w = w_x[i][nan_mask]
            real_w = real_w.reshape(w_x.shape[1], z_ind)

            if new_formula:
                z_ind = min(z_ind, z_ind_next)
            du_dt = np.zeros((steps, z_ind), dtype=np.float64)
            du = np.diff(real_u, axis=0)
            du_dt[0:steps - 1, :] = du[:, 0:z_ind] / self.dt
            du_dx = np.zeros((steps, z_ind), dtype=np.float64)
            du_dz = np.zeros((steps, z_ind), dtype=np.float64)
            if new_formula:
                dx = self.x[x_inds[i] + 1] - self.x[x_inds[i]]
                du_dx[:, :] = (real_u_next[:, 0:z_ind] - real_u[:, 0:z_ind]) / dx
                du = np.diff(real_u, axis=1)
                for z in range(0, z_ind - 1):
                    du_dz[:, z] = du[:, z] / np.abs(self.z[z] - self.z[z + 1])

            if new_formula:
                # new Folmula:
                # ??? g
                fn_drag = real_rho[:, 0:z_ind] * R * Cd * real_u[:, 0:z_ind] * abs(real_u[:, 0:z_ind])
                fn_inertia = real_rho[:, 0:z_ind] * R * R * np.pi * Cm * real_u[:, 0:z_ind] * du_dx
                if with_sec_part:
                    fn_inertia += real_rho[:, 0:z_ind] * R * R * np.pi * Cm * real_w[:, 0:z_ind] * du_dz
            else:
                # Morison Formula:
                fn_drag = real_rho * R * Cd * real_u * abs(real_u)
                fn_inertia = real_rho * R * np.pi * R * Cm * du_dt
            fn = fn_drag + fn_inertia
            for z in range(0, z_ind):
                fn_d[i, :, z] = fn_drag[:, z]
                fn_i[i, :, z] = fn_inertia[:, z]
            fn_z = fn * self.z[0:z_ind]
            for t in range(0, steps):
                F_n[i, t] = np.trapz(self.z[0:z_ind], fn[t, :])
                M[i, t] = np.trapz(self.z[0:z_ind], fn_z[t, :])
        return F_n, M, steps, fn_d, fn_i


def interpolate_field(x, y, data, x_factor=1, y_factor=1):
    f = interp1d(range(0, len(x)), x, kind='slinear')
    x_array = f(np.linspace(0, len(x) - 1, len(x) * x_factor))
    f = interp1d(range(0, len(y)), y, kind='slinear')
    y_array = f(np.linspace(0, len(y) - 1, len(y) * y_factor))
    X, Y = np.meshgrid(y_array, x_array)
    res = interpn((x, y), data, (Y, X), method='linear')
    return res, x_array, y_array
