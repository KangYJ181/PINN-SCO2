import numpy as np
import pandas as pd
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary


def physics(ldr, pre, dia, tem):
    # Edit the physical constraint P2
    # ldr: length to diameter ratio
    # pre: pressure (MPa)
    # dia: diameter (mm)
    # tem: temperature (K)
    # rou: density (kg/m3)
    # w: mass flow rate (g/s)
    P = pre * 1000000  # Convert the pressure unit to Pa so that REFPROP can use it
    z = [1.0]  # For pure fluid, to call density for REFPROP
    rou = np.zeros_like(pre)  # Initialize density array
    w = np.zeros_like(pre)  # Initialize mass flow rate array

    for i in range(len(pre)):
        result = RP.REFPROPdll('CO2', 'PT', 'D', RP.MASS_BASE_SI, iMass, iFlag, P[i], tem[i], z)
        rou[i] = result.Output[0]
        w[i] = np.pi * (dia[i] / 2) ** 2 \
               * (0.5463 + 0.0587 * (pre[i] / 7.39) ** 2.07 * (rou[i] / 354.36) ** -0.939) \
               * (2 * rou[i] * pre[i] * abs(1 - (0.579 + 0.024 * np.log(ldr[i])))) ** 0.5
    return w


# Set the REFPROP
RP = REFPROPFunctionLibrary('C:/Program Files (x86)/REFPROP')
iMass = 1  # 1 represents the mass basis
iFlag = 0  # 0 represents the standard calculation process
MASS_BASE_SI = RP.GETENUMdll(iFlag, "MASS BASE SI").iEnum  # Only REFPROP 10.0 can use this function

data = pd.read_csv('experimental data D1.csv')  # Load the data
x = data.drop('mas', axis=1).values
y = data['mas'].values.reshape(-1, 1)
x_gen = x[0:]
y_gen = y[0:]

ldr = x_gen[:, 0]
pre = x_gen[:, 1]
dia = x_gen[:, 2]
tem = x_gen[:, 4]

physics_outputs = []
w = physics(ldr, pre, dia, tem)
w_ = pd.DataFrame(w, columns=['mass_flow_rate'])
w_.to_csv('predictions_D1P1.csv', index=False)