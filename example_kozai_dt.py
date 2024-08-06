import numpy as np
import kozaipy as kp
from kozaipy import triples
import pickle

if __name__ == "__main__":

    m1 = kp.constants.Rjup
    k2_1 = 0.25
    k2_0 = 0.014
    timelag0 = 1e-8
    timelag1 = 1 / (3600 * 24)
    R1 = kp.constants.Rsun / 10
    R0 = kp.constants.Rsun
    m0 = 1
    tv1 = (1.5 / timelag1) * (R1**3 / triples.constants.G) / m1 * (1 + 2 * k2_1) ** 2 / k2_1
    #tv0 = 1.5 / timelag0 * R0 * R0 * R0 / triples.constants.G / m0 * (1 + 2 * k2_0) ** 2 / k2_1
    print(tv1)

    trip = kp.Triple(m0=1, m1=1e-3, m2=1, a1=1.5, a2=225.0, e1=0.01, e2=1e-5, I=87* np.pi / 180.0,
                     type0='star', type1='planet',
                     g1=0, g2=360 * np.pi/180.0,
                     spin_rate0=2 * np.pi / 2.3, spin_rate1 = 2 * np.pi/0.417,
                     pseudosynch1=False,
                     spinorbit_align0=False, spinorbit_align1=True,
                     R0=kp.constants.Rsun, R1=kp.constants.Rsun / 10,
                     k2_0=0.014, k2_1=0.25, tv0=1e8, tv1=tv1, rg0=0.07, rg1=0.28, spin1=True)

    sol = trip.integrate(timemin=0.0, timemax=1e8 * 365.25, Nevals=120000,
                         octupole_potential=True,
                         short_range_forces_conservative=True,
                         short_range_forces_dissipative=True,
                         version='tides_dt')

    with open('high-dt-e-migration.pickle', 'wb') as file:
        pickle.dump(sol, file)

