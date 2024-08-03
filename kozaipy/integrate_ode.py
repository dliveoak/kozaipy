
import numpy as np
import scipy.integrate as integ
from matplotlib import pyplot as plt
from kozaipy import triples
from numpy import random
import pickle
from kozaipy.triples_integrate_tides import threebody_ode_vf_tides_modified


def integrate_callback(ode, ics, t0, tf, callback, args = [], echo_buffer = 100, int_kwargs = {}, map_params = {}):

    ts = [t0]
    y = [ics]
    amps = []

    dt = callback(ts, y, map_params)


    N = 0
    print('{:<6s}       {:<10s}   {:<10s}'.format('t', 'dt', 'finished'))
    while ts[-1] < tf:
        if N % echo_buffer == 0:
            print('{:<6f} Myr   {:<10f}   {:<10f}%'.format(ts[-1] / (365.25 * 1e6), dt, 100 * ts[-1] / tf),
                  str(map_params['amplitude_last']), str(map_params['dP']), str(map_params['da']), str(map_params['P']), str(map_params['E_B_tilde_last']))
        amps.append(map_params['amplitude_last'])
        ts.append(ts[-1] + dt)
        sol = integ.odeint(ode, y[-1], [ts[-2], ts[-1]], args=args, **int_kwargs)
        y.append(sol[-1][:])
        dt = callback(ts, y, map_params)
        N += 1

    ts = np.array(ts)
    y = np.array(y)
    amps = np.array(amps)

    return ts, y, amps


if __name__ == '__main__':

    pass
    # def callback(params):
    #     dt = 200*365.25
    #     return dt
    #
    #
    # with open('../scraped.pickle', 'rb') as file:
    #     dict = pickle.load(file)
    #     print(dict)
    #     print(dict['ics'])
    #
    # triples.triple_data = {'m0': False, 'm1': False, 'm2': False, 'inner_orbit': True, 'outer_orbit': True, 'spin0': True, 'spin1': False, 'spinorbit_align0': False, 'spinorbit_align1': False, 'pseudosynch0': False, 'pseudosynch1': True}
    #
    # t, sol = integrate_callback(threebody_ode_vf_tides_modified,
    #                    ics=dict['ics'], t0=0, tf=365.25*1e9, callback=callback, args=dict['args'], echo_buffer=1000)
    #
    # np.savetxt('vick-t.csv', t)
    # np.savetxt('vick-orbits.csv', sol)
