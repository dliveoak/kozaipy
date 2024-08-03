import numpy as np
import scipy.integrate as integ
import kozaipy.triples as triples
import kozaipy as kp
from kozaipy.triples_integrate_full import *
from kozaipy.triples_integrate_tides import *
from kozaipy.triples_integrate_single_average import *
from kozaipy.integrate_ode import integrate_callback
import pickle

# import bsint


triple_precision = {"e1x": 1.0e-8,
                    "e1y": 1.0e-8,
                    "e1z": 1.0e-8,
                    "l1x": 1.0e-10,
                    "l1y": 1.0e-10,
                    "l1z": 1.0e-10,
                    "e2x": 1.0e-6,
                    "e2y": 1.0e-6,
                    "e2z": 1.0e-6,
                    "l2x": 1.0e-8,
                    "l2y": 1.0e-8,
                    "l2z": 1.0e-8,
                    "spin0x": 1.0e-8,
                    "spin0y": 1.0e-8,
                    "spin0z": 1.0e-8,
                    "spin1x": 1.0e-7,
                    "spin1y": 1.0e-7,
                    "spin1z": 1.0e-7,
                    "Omega0": 1.0e-7,
                    "Omega0x": 1.0e-7,
                    "Omega0y": 1.0e-7,
                    "Omega0z": 1.0e-7,
                    "Omega1": 1.0e-5,
                    "Omega1x": 1.0e-5,
                    "Omega1y": 1.0e-5,
                    "Omega1z": 1.0e-5,
                    "m0": 1e-8,
                    "m1": 1e-8,
                    "m2": 1e-8,
                    "R0": 1e-10,
                    "R1": 1e-10}


def integrate_triple_system(ics, timemin, timemax, Nevals,
                            body0, body1, body2,
                            octupole_potential=True,
                            short_range_forces_conservative=False,
                            short_range_forces_dissipative=False,
                            solve_for_spin_vector=False,
                            version='tides',
                            tol=1.0e-10,
                            integrator='scipy',
                            add_derivatives=False):
    atol = tol
    rtol = tol / 10.0
    rtol = 1.0e-8

    m0, m1, m2 = body0.mass, body1.mass, body2.mass
    radius0, radius1 = body0.radius, body1.radius
    dradius0_dt, dradius1_dt = body0.dradius_dt, body1.dradius_dt
    gyroradius0, gyroradius1 = body0.gyroradius, body1.gyroradius
    dgyroradius0_dt, dgyroradius1_dt = body0.dgyroradius_dt, body1.dgyroradius_dt
    k2_0, k2_1 = body0.apsidal_constant, body1.apsidal_constant
    tv0, tv1 = body0.viscous_time, body1.viscous_time
    tauconv0, tauconv1 = body0.convective_time, body1.convective_time
    tlag0, tlag1 = body0.tidal_lag_time, body1.tidal_lag_time

    rtol, atol = np.zeros(len(ics)), np.zeros(len(ics))
    for key, val in triples.triple_keys.items():
        if val is not None:
            rtol[val] = 10 * triple_precision[key]
            atol[val] = triple_precision[key]

    # rtol,atol = 1.e-9,1.e-08

    # integrator='other'
    # integrator = 'bsint'

    time = np.linspace(timemin, timemax, Nevals)
    params = m0, m1, m2, radius0, radius1, gyroradius0, gyroradius1, k2_0, k2_1, \
        tv0, tv1, tauconv0, tauconv1, \
        tlag0, tlag1, \
        dradius0_dt, dradius1_dt, \
        dgyroradius0_dt, dgyroradius1_dt, \
        octupole_potential, \
        short_range_forces_conservative, short_range_forces_dissipative, solve_for_spin_vector

    if (integrator == 'scipy'):
        if (version == 'tides'):

            params = m0, m1, m2, radius0, radius1, gyroradius0, gyroradius1, k2_0, k2_1, tv0, tv1, \
                octupole_potential, \
                short_range_forces_conservative, short_range_forces_dissipative, solve_for_spin_vector

            with open('scraped.pickle', 'wb') as file:
                data = {'ics': ics, 'time': time, 'args': params, 'atol': atol,
                        'rtol': rtol, 'mxstep': 1000000, 'hmin': 0.0000001, 'mxords': 16, 'mxordn': 12}
                pickle.dump(data, file)

            print(triples.triple_data)

            sol = integ.odeint(threebody_ode_vf_tides_modified, ics, time, \
                               args=params, \
                               atol=atol, rtol=rtol, mxstep=1000000, hmin=0.0000001, mxords=16, mxordn=12)

        elif version == 'tides_dt':
            params = m0, m1, m2, radius0, radius1, gyroradius0, gyroradius1, k2_0, k2_1, tv0, tv1, \
                octupole_potential, \
                short_range_forces_conservative, short_range_forces_dissipative, solve_for_spin_vector

            def get_T(Q, z, eta, sigma, epsilon):
                K = 2 * z ** (3 / 2) * eta ** (3 / 2) * np.exp(-2 * z / 3) * (
                            1 - np.pi ** (1 / 2) / (4 * z ** (1 / 2))) / (15 ** (1 / 2))
                return 2 * np.pi ** 2 * Q ** 2 * K ** 2 * (sigma / epsilon)

            def get_omega_sigma_epsilon_Q(mode_index, Omega_bar, bar):
                if mode_index == 0:
                    return (1.22 - Omega_bar) * bar, (1.22 + Omega_bar) * bar, 1.22, 0.56
                if mode_index == 1:
                    return 0.56 * Omega_bar * bar, 2.56 * Omega_bar * bar, 0.28 * Omega_bar * bar, 0.015 * Omega_bar**2
                if mode_index == 2:
                    return -1.1 * Omega_bar * bar, 0.9 * Omega_bar * bar, -0.55 * Omega_bar * bar, 0.01 * Omega_bar**2
                if mode_index == 3:
                    return (1.46 - Omega_bar) * bar, (1.46 + Omega_bar) * bar, 1.46, 0.49

            def get_map_params(e, r_p, M, M_p, R, Omega, mode_index):
                r_tide = R * (M / M_p) ** (1 / 3)
                eta = r_p / r_tide
                Omega_peri = (kp.constants.G * (M + M_p) / r_p ** 3) ** (1 / 2)
                bar = (kp.constants.G * M_p / R**3)**(1/2)
                Omega_bar = Omega / bar

                #omega = (1.46 - Omega_bar) * bar
                #sigma = (1.46 + Omega_bar) * bar
                #epsilon = 1.46 * bar

                omega, sigma, epsilon, Q = get_omega_sigma_epsilon_Q(mode_index, Omega_bar, bar)

                z = 2 ** (1 / 2) * omega / Omega_peri
                T = get_T(Q=Q, z=z, eta=eta, sigma=sigma, epsilon=epsilon)

                dP = (6 * np.pi * (omega / Omega_peri) / (1 - e) ** (5 / 2)) * (M_p / M) ** (2 / 3) * eta ** (-5) * T
                P = 2 * np.pi * (omega / Omega_peri) / (1 - e) ** (3 / 2)

                dE = kp.constants.G * M**2 * R**5 * T / r_p**6

                return P, dP, dE

            def callback(ts, y, map_params):

                enable_dt = False

                e_squared = y[-1][0] ** 2 + y[-1][1] ** 2 + y[-1][2] ** 2
                e = e_squared**(1/2)
                l_squared = y[-1][3] ** 2 + y[-1][4] ** 2 + y[-1][5] ** 2
                a = l_squared / (kp.constants.G * (m0 + m1) * (1 - e_squared))
                P = (4 * np.pi ** 2 * a ** 3 / (kp.constants.G * (m0 + m1))) ** (1 / 2)
                Omega = sqrt(y[-1][6]**2 + y[-1][7]**2 + y[-1][8]**2)

                for i in [3]:

                    # new map params
                    P_hat_0, dP_hat_1, dE_inf = get_map_params(e=e, r_p=a * (1 - e), M=m0, M_p=m1, R=radius1, Omega=Omega, mode_index=i)
                    da = ((2 / 3) * dP_hat_1 / P_hat_0) ** (1 / 2)

                    amplitude_last = map_params['amplitude_last'][i]
                    E_tilde_last = map_params['E_tilde_last'][i]
                    E_bind_tilde = map_params['E_bind_tilde']
                    E_B_tilde_last = map_params['E_B_tilde_last'][i]

                    if dP_hat_1 >= 0.01:

                        enable_dt = True

                        # update map
                        dE_tilde = np.abs(amplitude_last + da) ** 2 - np.abs(amplitude_last) ** 2
                        EB0 = -kp.constants.G * m0 * m1 / (2 * a)
                        dE_tilde = -dE_inf / EB0
                        E_tilde = E_tilde_last + dE_tilde
                        E_B_tilde = E_B_tilde_last - dE_tilde

                        # if amplitude is sufficiently high, non-linear dissipation
                        if abs(amplitude_last) ** 2 >= 0.1 * E_bind_tilde:
                            print('Dissipate')
                            amplitude_last = (1e-3 * E_bind_tilde) ** (1 / 2) * (amplitude_last / abs(amplitude_last))
                            print(amplitude_last)
                            E_tilde_last = 1e-3 * E_bind_tilde

                        # new orbital period and amplitude
                        P_hat_new = P_hat_0 * (-1 / E_B_tilde) ** (3 / 2)
                        a_next = (amplitude_last + np.sqrt(dE_tilde+0j)) * np.exp(-complex(0, 1) * P_hat_new)

                        E_ratio = E_B_tilde_last / E_B_tilde

                        a_prime = E_ratio * a
                        e_prime = (1 - (1 / E_ratio) * (1 - e ** 2)) ** (1 / 2)
                        l_prime = np.sqrt(kp.constants.G * a_prime * (m0 + m1) * (1 - e_prime**2))

                        # update orbital elements in secular integration
                        y[-1][0:3] = np.array(y[-1][0:3]) * e_prime / e
                        y[-1][3:6] = np.array(y[-1][3:6]) * l_prime / l_squared**(1/2)

                        # update map params
                        E_B_tilde_last = E_B_tilde
                        E_tilde_last = E_tilde
                        amplitude_last = a_next

                        map_params['amplitude_last'][i] = amplitude_last
                        map_params['E_tilde_last'][i] = E_tilde_last
                        map_params['E_B_tilde_last'][i] = E_B_tilde_last
                        map_params['dP'][i] = dP_hat_1
                        map_params['P'][i] = P_hat_0
                        map_params['da'][i] = da

                        dt = P
                if not enable_dt:
                    dt = 1000*365.25
                return dt

            int_kwargs = {'atol': atol, 'rtol': rtol, 'mxstep': 1000000, 'hmin': 0.0000001, 'mxords': 16, 'mxordn': 12}
            map_params = {'amplitude_last' : np.zeros(4), 'E_tilde_last' : np.zeros(4), 'E_B_tilde_last' : -1 * np.ones(4),
                          'E_bind_tilde' : 2 * m1 * 1.5 / (radius1 * m0), 'dP' : np.zeros(4), 'P' : np.ones(4), 'da' : np.zeros(4)}

            time, sol, amps = integrate_callback(threebody_ode_vf_tides_modified, ics, time[0], time[-1],
                                           callback=callback,
                                           args=params,
                                           int_kwargs=int_kwargs,
                                           map_params=map_params)


            with open('amplitudes.pickle', 'wb') as file:
                pickle.dump(amps, file)

        elif (version == 'full'):
            sol = integ.odeint(threebody_ode_vf_full_modified, ics, time,
                               args=params,
                               atol=atol, rtol=rtol, mxstep=1000000, hmin=0.000000001, mxords=12, mxordn=10)
            if (add_derivatives):
                for kk, y in enumerate(sol):
                    if (kk == 0):
                        dsol_dt = threebody_ode_vf_full_modified(y, time[kk], *params)[
                                  :len(triples.triple_derivative_keys)]
                    else:
                        dsol_dt = np.vstack((dsol_dt, threebody_ode_vf_full_modified(y, time[kk], *params)[
                                                      :len(triples.triple_derivative_keys)]))


    else:
        if (version == 'tides'):
            params = [p for p in params if p is not None]
            solver = integ.ode(threebody_ode_vf_tides).set_integrator('dopri', nsteps=3000000, atol=atol, rtol=rtol,
                                                                      method='bdf')
        elif (version == 'full'):
            # solver = integ.ode(threebody_ode_vf_full).set_integrator('lsoda',nsteps=3000000,atol=atol,rtol=rtol, method='bdf',min_step=0.00001,max_order_ns=10)
            solver = integ.ode(threebody_ode_vf_full).set_integrator('dopri', nsteps=3000000, atol=atol, rtol=rtol,
                                                                     method='bdf')

        solver.set_initial_value(ics, time[0]).set_f_params(*params)
        kk = 1
        sol = []
        sol.append(ics)
        while solver.successful() and solver.t < time[-1]:
            print(100 * time[kk] / time[-1])
            solver.integrate(time[kk])
            sol.append(solver.y)
            kk += 1
        sol = np.asarray(sol)

    retval = np.column_stack((time, sol))

    if add_derivatives: retval = retval, dsol_dt

    return retval


def integrate_triple_system_sa(ics, timemin, timemax, Nevals,
                               body0, body1, body2,
                               octupole_potential=True,
                               short_range_forces_conservative=False,
                               short_range_forces_dissipative=False,
                               solve_for_spin_vector=False,
                               fix_outer_orbit=True,
                               tol=1.0e-10):
    if (fix_outer_orbit):
        # turn off integration of external orbit
        triples.triple_keys['e2x'], triples.triple_keys['e2y'], triples.triple_keys['e2z'] = None, None, None
        triples.triple_keys['l2x'], triples.triple_keys['l2y'], triples.triple_keys['l2z'] = None, None, None

    m0, m1, m2 = body0.mass, body1.mass, body2.mass
    radius0, radius1 = body0.radius, body1.radius
    dradius0_dt, dradius1_dt = body0.dradius_dt, body1.dradius_dt
    gyroradius0, gyroradius1 = body0.gyroradius, body1.gyroradius
    dgyroradius0_dt, dgyroradius1_dt = body0.dgyroradius_dt, body1.dgyroradius_dt
    k2_0, k2_1 = body0.apsidal_constant, body1.apsidal_constant
    tv0, tv1 = body0.viscous_time, body1.viscous_time
    tauconv0, tauconv1 = body0.convective_time, body1.convective_time
    tlag0, tlag1 = body0.tidal_lag_time, body1.tidal_lag_time

    rtol, atol = np.zeros(len(ics)), np.zeros(len(ics))
    for key, val in triples.triple_keys.items():
        if val is not None:
            rtol[val] = 10 * triple_precision[key]
            atol[val] = triple_precision[key]

    params = m0, m1, m2, radius0, radius1, gyroradius0, gyroradius1, k2_0, k2_1, \
        tv0, tv1, tauconv0, tauconv1, \
        tlag0, tlag1, \
        dradius0_dt, dradius1_dt, \
        dgyroradius0_dt, dgyroradius1_dt, \
        octupole_potential, \
        short_range_forces_conservative, short_range_forces_dissipative, solve_for_spin_vector

    if (fix_outer_orbit):
        new_params = ics[6:12]
        ics = np.append(ics[0:6], ics[12:])
        rtol = np.append(rtol[0:6], rtol[12:])
        atol = np.append(atol[0:6], atol[12:])
        for p in params: new_params.append(p)
        params = tuple(new_params)

    time = np.linspace(timemin, timemax, Nevals)

    sol = integ.odeint(threebody_ode_vf_sa, ics, time, \
                       args=params, \
                       mxstep=1000000, hmin=0.000000001, mxords=12, mxordn=10)
    # atol=atol,rtol=rtol)#,

    retval = np.column_stack((time, sol))

    return retval
