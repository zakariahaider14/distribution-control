import numpy as np
import pandas as pd
import pyomo.environ as pyo
from copy import deepcopy
import os

# Parameters
T = 10       # Total time steps
N = 3        # MPC horizon
n = 34       # Number of bus nodes
DER_buses = [5, 8, 10]
Cap_banks = [15, 18]
DER_Pmax = 0.5
DER_Qmax = 0.3
Cap_kVAR = 0.2

# Load Ybus
Ybus = pd.read_csv("Ybus_matrix.csv", index_col=0).values
G = Ybus.real
B = Ybus.imag

# Logging structures
log_voltages = []
log_DER = []
log_caps = []

# Initialize true voltage state
Vr_true = np.random.normal(1.0, 0.01, size=n)
Vi_true = np.random.normal(0.0, 0.01, size=n)

# Function to simulate noisy measurements of |V|
def simulate_measurement(Vr, Vi, noise_std=0.005):
    return np.sqrt(Vr**2 + Vi**2) + np.random.normal(0, noise_std, size=n)

# WLS estimation function using Gauss-Newton method
def wls_voltage_magnitude_only(V_meas, iterations=5, noise_std=0.005):
    Vr_est = np.ones(n)
    Vi_est = np.zeros(n)
    for _ in range(iterations):
        Vmag_est = np.sqrt(Vr_est**2 + Vi_est**2)
        H = np.zeros((n, 2 * n))
        for i in range(n):
            denom = max(Vmag_est[i], 1e-6)
            H[i, i] = Vr_est[i] / denom
            H[i, i + n] = Vi_est[i] / denom
        r = V_meas - Vmag_est
        dx = np.linalg.lstsq(H, r, rcond=None)[0]
        Vr_est += dx[:n]
        Vi_est += dx[n:]
    return Vr_est, Vi_est

for t in range(T):
    print(f"\n--- Time Step {t} ---")

    # Simulate measurement from true state
    V_meas = simulate_measurement(Vr_true, Vi_true)

    # Estimate state from measurement using WLS
    Vr_est, Vi_est = wls_voltage_magnitude_only(V_meas)

    # Pyomo model setup
    model = pyo.ConcreteModel()
    model.t = pyo.RangeSet(0, N - 1)
    model.BUS = pyo.RangeSet(0, n - 1)
    model.Vr = pyo.Var(model.t, model.BUS, bounds=(-1.1, 1.1))
    model.Vi = pyo.Var(model.t, model.BUS, bounds=(-1.1, 1.1))
    model.P_DER = pyo.Var(model.t, DER_buses, bounds=(0, DER_Pmax))
    model.Q_DER = pyo.Var(model.t, DER_buses, bounds=(-DER_Qmax, DER_Qmax))
    model.Cap = pyo.Var(model.t, Cap_banks, within=pyo.Binary)

    def obj_rule(m):
        dev = sum((m.Vr[t, i]**2 + m.Vi[t, i]**2 - 1)**2 for t in m.t for i in m.BUS)
        der = sum((DER_Pmax - m.P_DER[t, i])**2 for t in m.t for i in DER_buses)
        cap = sum(m.Cap[t, i] for t in m.t for i in Cap_banks)
        return dev + 0.1 * der + 0.01 * cap
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def voltage_limits(m, t, i):
        return pyo.inequality(0.95**2, m.Vr[t, i]**2 + m.Vi[t, i]**2, 1.05**2)
    model.Vlimit = pyo.Constraint(model.t, model.BUS, rule=voltage_limits)

    def balance_P(m, t, i):
        Ir = sum(G[i, j]*m.Vr[t, j] - B[i, j]*m.Vi[t, j] for j in m.BUS)
        Ii = sum(B[i, j]*m.Vr[t, j] + G[i, j]*m.Vi[t, j] for j in m.BUS)
        Pi = m.Vr[t, i]*Ir + m.Vi[t, i]*Ii
        if i in DER_buses:
            Pi -= m.P_DER[t, i]
        return Pi == 0
    model.Pbal = pyo.Constraint(model.t, model.BUS, rule=balance_P)

    def balance_Q(m, t, i):
        Ir = sum(G[i, j]*m.Vr[t, j] - B[i, j]*m.Vi[t, j] for j in m.BUS)
        Ii = sum(B[i, j]*m.Vr[t, j] + G[i, j]*m.Vi[t, j] for j in m.BUS)
        Qi = m.Vi[t, i]*Ir - m.Vr[t, i]*Ii
        if i in DER_buses:
            Qi -= m.Q_DER[t, i]
        if i in Cap_banks:
            Qi -= Cap_kVAR * m.Cap[t, i]
        return Qi == 0
    model.Qbal = pyo.Constraint(model.t, model.BUS, rule=balance_Q)

    for i in range(n):
        model.Vr[0, i].fix(Vr_est[i])
        model.Vi[0, i].fix(Vi_est[i])

    solver = pyo.SolverFactory("ipopt")
    solver.solve(model)

    # Apply first control step
    Vr_true = np.array([pyo.value(model.Vr[1, i]) for i in model.BUS])
    Vi_true = np.array([pyo.value(model.Vi[1, i]) for i in model.BUS])
    P_der_t = {i: pyo.value(model.P_DER[0, i]) for i in DER_buses}
    Q_der_t = {i: pyo.value(model.Q_DER[0, i]) for i in DER_buses}
    Cap_t = {i: int(pyo.value(model.Cap[0, i])) for i in Cap_banks}

    log_voltages.append((Vr_true.copy(), Vi_true.copy()))
    log_DER.append((P_der_t, Q_der_t))
    log_caps.append(Cap_t)

# Save
os.makedirs("mpc_output", exist_ok=True)
pd.DataFrame(np.array(log_voltages).reshape(T, -1)).to_csv("mpc_output/voltages.csv")
pd.DataFrame(log_DER).to_csv("mpc_output/DER_controls.csv")
pd.DataFrame(log_caps).to_csv("mpc_output/Capacitor_states.csv")
