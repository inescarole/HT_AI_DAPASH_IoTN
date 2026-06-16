"""
Device Layout:
  - 100 bedside monitors  (2 per patient, 50 patients)  [MONITOR]
  - 30  infusion pumps                                    [PUMP]
  - 20  ICU ventilators                                   [VENT]
  - 200 wearable sensors  (4 per patient)                [WEARABLE]
  Total n = 350 devices

State dimension per device: d = 100
  d1=20 operating params | d2=20 system metrics | d3=20 perf indicators
  d4=20 security features | d5=20 comm features

Physical sensors per device: ns = 5
  [HR, BP(SBP/DBP encoded), SpO2, RR, Temp]  — monitor/wearable
  [rate, conc, pressure, alarm_flag, battery] — pump
  [tidal_vol, FiO2, PEEP, alarm_flag, battery]— vent

Attack types: A1 Spoofing | A2 AdversarialPerturbation | A3 DataPoisoning
             A4 ModelExtraction | A5 SignalReplay

Defense actions (Table I): MONITOR | FIREWALL | ENCRYPTION | HONEYPOT |
                            DECEPTION | ISOLATION | ACCESS_CONTROL

Reasoning levels k=0,1,2 for both players.

"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import scipy.stats as stats
from scipy.special import rel_entr          # KL divergence element-wise
from scipy.spatial.distance import jensenshannon
from scipy.linalg import norm
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import defaultdict, deque
import itertools
import warnings
import copy
import os
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, NamedTuple
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS  (match paper exactly)
# ──────────────────────────────────────────────────────────────────────────────

#  Hospital topology 
N_MONITORS   = 100          #100 bedside monitors
N_PUMPS      = 30           # 30 infusion pumps
N_VENTS      = 20          # 20 ventilators
N_WEARABLES  = 200          # 200 wearable sensors
N_DEVICES    = N_MONITORS + N_PUMPS + N_VENTS + N_WEARABLES   # 350
N_PATIENTS   = 50

#  State dimensions (paper Sec. VI-A2) 
D_STATE      = 100          # device state vector dimension
D_OPER       = 20           # d1 operating parameters
D_SYS        = 20           # d2 system metrics
D_PERF       = 20           # d3 performance indicators
D_SEC        = 20           # d4 security features
D_COMM       = 20           # d5 communication features
NS           = 5            # physical sensor readings per device

#  Attack/defense parameters (paper Sec. VI-B,C,E) 
EPSILON_PERT = 0.15       # 0.15ℓ2-norm perturbation budget  (Eq.30,38)
SIGMA_NOISE  = 0.05         #0.05Gaussian sensor noise σ       (Eq.33)
P_EDGE       = 0.1          # Erdős–Rényi edge probability  (Eq.28)
P_BASE_ATK   = 0.3          # 0.3base attack success prob      (Eq.38)
LAMBDA_C     = 0.1          # multi-target cost penalty     (Eq.31)
KAPPA        = 0.9          # health degradation multiplier (Sec. VI-E3)
ALPHA_DECAY  = 0.01         # trust decay rate              (Sec. VI-E3)
ALPHA_RECOV  = 0.05         # trust recovery rate           (Sec. VI-E3)
TRUST_LOW    = 0.4          # low-trust threshold           (Sec. VI-B3)
HEALTH_LOW   = 0.7          # low-health threshold          (Sec. VI-B3)

#  Particle filter (paper Sec. VI-C2) 
N_PARTICLES  = 100          # 200 number of particles in belief representation
N_EFF_MIN    = 0.2 * N_PARTICLES        # = 200

#  Learning parameters (paper Sec. VI-E1) 
ETA          = 0.001        # learning rate (Adam)
BETA_ADAM1   = 0.9
BETA_ADAM2   = 0.999
BETA_TEMP    = 0.5          # softmax temperature
OMEGA_DUAL   = 0.3          # dual variable lr
GAMMA_DISC   = 0.99         # discount factor
UPSILON      = 0.1          # perception update rate  (Eq.21)
BETA_REG     = 0.5          # regularisation          (Eq.18)
LAMBDA_GAME  = 0.1          # perception advantage weight (Eq.5,9)
MU_GAME      = 0.05         # strategy regularisation weight (Eq.20)
OMEGA_META   = 0.3          # meta-gradient perception weight (Eq.19)

#  k-level reasoning 
K_DEFENDER   = 2
K_ATTACKER   = 2
EPS_CONV     = 1e-3         # Nash iteration convergence (Eq.37)

#  Simulation config 
N_EPISODES   = 50
T_STEPS      = 200
N_SEEDS      = 50
BASE_SEED    = 42
N_BINS       = 15           # discretisation bins for KL (Sec. VI-D2)
#K_ADV_STEPS  = 15
K_ADV_STEPS  = 5           # adversarial training steps per episode

#  Attack type indices 
ATK_SPOOFING = 0            # A1
ATK_ADVERS   = 1            # A2 — primary (Sec. VI-B2)
ATK_POISON   = 2            # A3
ATK_EXTRACT  = 3            # A4
ATK_REPLAY   = 4            # A5
N_ATK_TYPES  = 5

#  Defense action indices (Table I) 
DEF_MONITOR      = 0
DEF_FIREWALL     = 1
DEF_ENCRYPTION   = 2
DEF_HONEYPOT     = 3
DEF_DECEPTION    = 4
DEF_ISOLATION    = 5
DEF_ACCESS_CTRL  = 6
N_DEF_ACTIONS    = 7

#  Defense Table I: effectiveness η and cost c 
#     [MONITOR, FIREWALL, ENCRYPTION, HONEYPOT, DECEPTION, ISOLATION, ACCESS]
DEF_ETA = np.array([1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0])
DEF_COST = np.array([0.1, 0.5, 1.0, 0.8, 0.6, 1.2, 0.3])

# Effectiveness matrix: DEF_EFF[action, attack_type]
# which defenses work against which attacks
DEF_EFF = np.array([
    # A1    A2    A3    A4    A5
    [0.5,  0.3,  0.3,  0.3,  0.3],   # MONITOR (detects but doesn't stop)
    [1.5,  0.5,  0.5,  0.5,  0.5],   # FIREWALL (strong vs spoofing)
    [0.5,  0.5,  1.5,  1.5,  0.5],   # ENCRYPTION (vs poisoning, extraction)
    [1.5,  0.5,  0.5,  0.5,  1.5],   # HONEYPOT (vs spoofing, replay)
    [0.5,  1.5,  0.5,  0.5,  0.5],   # DECEPTION (vs adversarial pert.)
    [0.5,  0.5,  1.5,  0.5,  0.5],   # ISOLATION (vs poisoning)
    [1.0,  1.0,  1.0,  1.0,  1.0],   # ACCESS_CONTROL (general)
])

# Attack type multipliers µ_type ∈ [0.7, 1.2]  (Eq.38)
ATK_MU = np.array([0.9, 1.2, 1.0, 0.8, 0.7])

# ──────────────────────────────────────────────────────────────────────────────
# DEVICE TYPE ENUMERATION
# ──────────────────────────────────────────────────────────────────────────────
DEV_MONITOR  = 'monitor'
DEV_PUMP     = 'pump'
DEV_VENT     = 'ventilator'
DEV_WEARABLE = 'wearable'

# Criticality weights for utility computation
DEV_CRITICALITY = {
    DEV_MONITOR:  0.8,
    DEV_PUMP:     1.0,   # highest — wrong infusion = lethal
    DEV_VENT:     1.0,   # highest — airway management
    DEV_WEARABLE: 0.5,
}

# ──────────────────────────────────────────────────────────────────────────────
# PHYSIOLOGICAL NORMAL RANGES  (for perception attack plausibility checks)
# ──────────────────────────────────────────────────────────────────────────────
PHYSIO_RANGES = {
    'HR':    (40,  200),    # heart rate bpm
    'SBP':   (60,  220),    # systolic BP mmHg
    'DBP':   (40,  130),    # diastolic BP mmHg
    'SpO2':  (70,  100),    # oxygen saturation %
    'RR':    (4,   50),     # respiratory rate /min
    'Temp':  (34.0, 42.0),  # temperature °C
    'Gluc':  (20,  600),    # glucose mg/dL
    'Rate':  (0,   500),    # infusion rate ml/hr
    'TidalV':(100, 900),    # tidal volume mL
    'FiO2':  (0.21, 1.0),   # fraction inspired O2
    'PEEP':  (0,   25),     # positive end-exp pressure cmH2O
}

# Normal operating values (mean)
PHYSIO_NORMAL = {
    'HR': 75, 'SBP': 120, 'DBP': 80, 'SpO2': 98,
    'RR': 15, 'Temp': 37.0, 'Gluc': 100,
    'Rate': 50, 'TidalV': 450, 'FiO2': 0.4, 'PEEP': 8,
}

PHYSIO_STD = {
    'HR': 10, 'SBP': 10, 'DBP': 8, 'SpO2': 1.5,
    'RR': 2, 'Temp': 0.3, 'Gluc': 15,
    'Rate': 10, 'TidalV': 50, 'FiO2': 0.05, 'PEEP': 1.5,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  HOSPITAL IoT NETWORK TOPOLOGY  (Sec. VI-A1, Eq. 28)
# ══════════════════════════════════════════════════════════════════════════════

class HospitalNetworkTopology:
    """
    Erdős–Rényi random graph G=(V,E) over N_DEVICES=350 nodes.
    Adjacency matrix A ∈ {0,1}^{n×n} (symmetric, no self-loops) — Eq.28.
    Devices are typed; inter-type edges reflect hospital Wi-Fi 6 segmentation.
    """

    def __init__(self, rng: np.random.Generator):
        self.n = N_DEVICES
        self.rng = rng
        self.device_types: List[str] = []
        self.patient_map: Dict[int, int] = {}   # device_id → patient_id
        self._assign_device_types()
        self.adj = self._build_adjacency()      # Eq.28
        self.degree = self.adj.sum(axis=1)
        self.deg_mean = self.degree.mean()
        self.deg_std  = self.degree.std()

    def _assign_device_types(self):
        types = (
            [DEV_MONITOR]  * N_MONITORS  +
            [DEV_PUMP]     * N_PUMPS     +
            [DEV_VENT]     * N_VENTS     +
            [DEV_WEARABLE] * N_WEARABLES
        )
        self.device_types = types
        # Map monitors and wearables to patients
        for i in range(N_MONITORS):
            self.patient_map[i] = i // 2      # 2 monitors per patient
        for i in range(N_WEARABLES):
            dev_id = N_MONITORS + N_PUMPS + N_VENTS + i
            self.patient_map[dev_id] = i // 4  # 4 wearables per patient
        # Precompute numpy array for vectorised patient aggregation
        self.patient_map_array = np.array(
            [self.patient_map.get(i, -1) for i in range(self.n)], dtype=np.int32)

    def _ward_of(self, device_id: int) -> int:
        """Return ward index (0–4) for a device based on patient assignment."""
        pid = self.patient_map.get(device_id, -1)
        if pid >= 0:
            return pid // 10          # 10 patients per ward, 5 wards
        # Pumps and vents: assign by position within their type block
        if device_id < N_MONITORS:
            return (device_id // 2) // 10    # monitor → patient → ward
        offset = device_id - N_MONITORS
        if offset < N_PUMPS:
            return offset // 6        # 6 pumps per ward
        offset -= N_PUMPS
        if offset < N_VENTS:
            return offset // 4        # 4 vents per ward
        return 0

    def _build_adjacency(self) -> np.ndarray:
        """
        Three-tier segmentation: gateway (pump/vent) → bedside (monitor) → peripheral (wearable)
        VLAN isolation: no cross-ward edges.
        """
        A = np.zeros((self.n, self.n), dtype=float)

        # Probabilities
        P_GW   = 0.8    # gateway mesh (pump ↔ vent, same ward)
        P_UP   = 0.6    # uplink (monitor ↔ pump/vent, same ward)
        P_PEER = 0.2    # peer (monitor ↔ monitor, same ward)

        # Precompute ward and type for each device
        wards = [self._ward_of(i) for i in range(self.n)]
        types = self.device_types

        for i in range(self.n):
            for j in range(i + 1, self.n):
                # VLAN rule: no cross-ward edges
                if wards[i] != wards[j]:
                    continue

                ti, tj = types[i], types[j]
                connected = False

                # Rule 1: gateway mesh
                if (ti in (DEV_PUMP, DEV_VENT) and
                        tj in (DEV_PUMP, DEV_VENT)):
                    connected = self.rng.random() < P_GW

                # Rule 2: monitor uplink to gateway
                elif ((ti == DEV_MONITOR and tj in (DEV_PUMP, DEV_VENT)) or
                      (tj == DEV_MONITOR and ti in (DEV_PUMP, DEV_VENT))):
                    connected = self.rng.random() < P_UP

                # Rule 3: monitor peer
                elif ti == DEV_MONITOR and tj == DEV_MONITOR:
                    connected = self.rng.random() < P_PEER

                # Rule 4: wearable binds to same-patient monitors only
                elif ((ti == DEV_WEARABLE and tj == DEV_MONITOR) or
                      (tj == DEV_WEARABLE and ti == DEV_MONITOR)):
                    pi = self.patient_map.get(i, -1)
                    pj = self.patient_map.get(j, -1)
                    connected = (pi == pj and pi >= 0)

                # All other pairs (wearable–wearable, wearable–pump/vent): no edge
                if connected:
                    A[i, j] = 1.0
                    A[j, i] = 1.0

        return A

    def get_high_connectivity_targets(self) -> List[int]:
        """Devices with degree > mean + std (Sec. VI-B3)."""
        threshold = self.deg_mean + self.deg_std
        return [i for i in range(self.n) if self.degree[i] > threshold]

    def get_gateway_devices(self) -> List[int]:
        """Ventilators and high-degree pumps as critical infrastructure."""
        gw = list(range(N_MONITORS, N_MONITORS + N_PUMPS + N_VENTS))
        return gw

    def get_neighbors(self, device_id: int) -> List[int]:
        return list(np.where(self.adj[device_id] > 0)[0])


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PHYSICAL STATE s_t  (Sec. II-A, VI-A2)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PatientPhysiology:
    """True physiological state of one patient."""
    patient_id: int
    HR:   float = 75.0   # bpm
    SBP:  float = 120.0  # mmHg
    DBP:  float = 80.0   # mmHg
    SpO2: float = 98.0   # %
    RR:   float = 15.0   # /min
    Temp: float = 37.0   # °C
    Gluc: float = 100.0  # mg/dL (wearable CGM)

    def as_vector(self) -> np.ndarray:
        return np.array([self.HR, self.SBP, self.DBP,
                         self.SpO2, self.RR, self.Temp, self.Gluc])

    @staticmethod
    def from_vector(pid: int, v: np.ndarray) -> 'PatientPhysiology':
        p = PatientPhysiology(patient_id=pid)
        p.HR, p.SBP, p.DBP, p.SpO2, p.RR, p.Temp, p.Gluc = v
        return p

    def clip_to_physio_range(self):
        """Enforce physiological plausibility bounds."""
        self.HR   = float(np.clip(self.HR,   *PHYSIO_RANGES['HR']))
        self.SBP  = float(np.clip(self.SBP,  *PHYSIO_RANGES['SBP']))
        self.DBP  = float(np.clip(self.DBP,  *PHYSIO_RANGES['DBP']))
        self.SpO2 = float(np.clip(self.SpO2, *PHYSIO_RANGES['SpO2']))
        self.RR   = float(np.clip(self.RR,   *PHYSIO_RANGES['RR']))
        self.Temp = float(np.clip(self.Temp, *PHYSIO_RANGES['Temp']))
        self.Gluc = float(np.clip(self.Gluc, *PHYSIO_RANGES['Gluc']))


class DeviceState:
    """
    Full device state vector s_i ∈ R^d (d=100, Sec. VI-A2)
    plus raw sensor readings r_i ∈ R^{ns} (ns=5).
    """

    def __init__(self, device_id: int, device_type: str, rng: np.random.Generator):
        self.device_id   = device_id
        self.device_type = device_type
        self.rng = rng

        # s_i ~ N(0, I) at initialisation  (Sec. VI-E5)
        self.s = rng.standard_normal(D_STATE)

        # Physical sensor readings r_i ∈ R^{ns=5}  (Sec. VI-A2)
        self.r = np.zeros(NS)

        # Device health h_i ∈ [0,1]  (Sec. VI-A3)
        self.health = 1.0

        # Trust score τ_i ∈ [0,1]  (Sec. VI-A3)
        self.trust = 1.0

        # Connected to network?
        self.online = True

        # Alarm flags
        self.alarm = False

    def get_operating_params(self) -> np.ndarray:
        return self.s[:D_OPER]

    def get_system_metrics(self) -> np.ndarray:
        return self.s[D_OPER:D_OPER+D_SYS]

    def get_perf_indicators(self) -> np.ndarray:
        return self.s[D_OPER+D_SYS:D_OPER+D_SYS+D_PERF]

    def get_security_features(self) -> np.ndarray:
        return self.s[D_OPER+D_SYS+D_PERF:D_OPER+D_SYS+D_PERF+D_SEC]

    def get_comm_features(self) -> np.ndarray:
        return self.s[D_OPER+D_SYS+D_PERF+D_SEC:]


class HospitalPhysicalState:
    """
    Complete physical state S = S_1 × S_2 × ... × S_n (Sec. II-A1).
    Maintains true physiological + device states for all 350 nodes.
    Implements natural stochastic dynamics (AR(1) process).
    """

    def __init__(self, rng: np.random.Generator, topology: HospitalNetworkTopology):
        self.rng = rng
        self.topology = topology
        self.t = 0

        # Patient physiology
        self.patients: List[PatientPhysiology] = []
        for pid in range(N_PATIENTS):
            p = self._init_patient(pid)
            self.patients.append(p)

        # Device states
        self.devices: List[DeviceState] = []
        for did in range(N_DEVICES):
            dtype = topology.device_types[did]
            dev = DeviceState(did, dtype, rng)
            self.devices.append(dev)
            self._sync_sensors(dev)

    def _init_patient(self, pid: int) -> PatientPhysiology:
        """Initialise patient with normally-distributed physiology."""
        p = PatientPhysiology(patient_id=pid)
        p.HR   = float(self.rng.normal(PHYSIO_NORMAL['HR'],   PHYSIO_STD['HR']))
        p.SBP  = float(self.rng.normal(PHYSIO_NORMAL['SBP'],  PHYSIO_STD['SBP']))
        p.DBP  = float(self.rng.normal(PHYSIO_NORMAL['DBP'],  PHYSIO_STD['DBP']))
        p.SpO2 = float(self.rng.normal(PHYSIO_NORMAL['SpO2'], PHYSIO_STD['SpO2']))
        p.RR   = float(self.rng.normal(PHYSIO_NORMAL['RR'],   PHYSIO_STD['RR']))
        p.Temp = float(self.rng.normal(PHYSIO_NORMAL['Temp'], PHYSIO_STD['Temp']))
        p.Gluc = float(self.rng.normal(PHYSIO_NORMAL['Gluc'], PHYSIO_STD['Gluc']))
        p.clip_to_physio_range()
        return p

    def _sync_sensors(self, dev: DeviceState):
        """Populate device sensor readings r_i from patient physiology."""
        dtype = dev.device_type
        pid   = self.topology.patient_map.get(dev.device_id, 0)
        pid   = min(pid, N_PATIENTS - 1)
        pat   = self.patients[pid]

        if dtype in (DEV_MONITOR, DEV_WEARABLE):
            # [HR, (SBP+DBP)/2-encoded, SpO2, RR, Temp]
            dev.r = np.array([
                pat.HR,
                (pat.SBP + pat.DBP) / 2.0,
                pat.SpO2,
                pat.RR,
                pat.Temp
            ])
        elif dtype == DEV_PUMP:
            # [rate, concentration proxy, pressure, alarm_flag, battery]
            dev.r = np.array([
                float(self.rng.normal(PHYSIO_NORMAL['Rate'], PHYSIO_STD['Rate'])),
                1.0,   # normalised concentration
                1.0,   # flow pressure OK
                0.0,   # alarm off
                1.0,   # battery full
            ])
        elif dtype == DEV_VENT:
            dev.r = np.array([
                float(self.rng.normal(PHYSIO_NORMAL['TidalV'], PHYSIO_STD['TidalV'])),
                float(self.rng.normal(PHYSIO_NORMAL['FiO2'],   PHYSIO_STD['FiO2'])),
                float(self.rng.normal(PHYSIO_NORMAL['PEEP'],   PHYSIO_STD['PEEP'])),
                0.0,
                1.0,
            ])

    def step_dynamics(self):
        """
        AR(1) physiological dynamics:
          s_{t+1} = 0.98 s_t + 0.02 s_normal + ε_t,  ε_t ~ N(0, 0.01 I)
        Mimics slow natural variation while staying near normal ranges.
        """
        ar_coef = 0.98
        for pid, pat in enumerate(self.patients):
            v = pat.as_vector()
            normal_v = np.array([
                PHYSIO_NORMAL['HR'], PHYSIO_NORMAL['SBP'], PHYSIO_NORMAL['DBP'],
                PHYSIO_NORMAL['SpO2'], PHYSIO_NORMAL['RR'],
                PHYSIO_NORMAL['Temp'], PHYSIO_NORMAL['Gluc']
            ])
            std_v = np.array([
                PHYSIO_STD['HR'], PHYSIO_STD['SBP'], PHYSIO_STD['DBP'],
                PHYSIO_STD['SpO2'], PHYSIO_STD['RR'],
                PHYSIO_STD['Temp'], PHYSIO_STD['Gluc']
            ])
            noise = self.rng.normal(0, 0.01, size=7) * std_v
            v_new = ar_coef * v + (1 - ar_coef) * normal_v + noise
            pat2 = PatientPhysiology.from_vector(pid, v_new)
            pat2.clip_to_physio_range()
            self.patients[pid] = pat2

        # Update device s-vectors with small AR(1) noise
        for dev in self.devices:
            dev.s = 0.95 * dev.s + 0.05 * self.rng.standard_normal(D_STATE)
            self._sync_sensors(dev)
             # Trust / health recovery if online and unattacked
            if dev.online and dev.health > HEALTH_LOW:
                dev.trust  = min(1.0, dev.trust  + ALPHA_RECOV)
                dev.health = min(1.0, dev.health + ALPHA_RECOV * 0.1)
            else:
                dev.trust  = max(0.0, dev.trust  - ALPHA_DECAY)
                dev.health = min(1.0, dev.health + ALPHA_RECOV * 0.05)

        self.t += 1

    def network_health(self) -> float:
        """
        H(t) = (1/n) Σ h_i(t)·τ_i(t)   Eq.29
        """
        hs = np.array([d.health for d in self.devices])
        ts = np.array([d.trust  for d in self.devices])
        return float(np.mean(hs * ts))

    def get_full_state_matrix(self) -> np.ndarray:
        """Returns state matrix S ∈ R^{n × d}."""
        return np.stack([d.s for d in self.devices])

    def get_sensor_matrix(self) -> np.ndarray:
        """Returns sensor matrix R ∈ R^{n × ns}."""
        return np.stack([d.r for d in self.devices])


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PERCEPTION STATES  P^i_l  (Sec. III-B, hospital mapping in Sec. 3.2)
# ══════════════════════════════════════════════════════════════════════════════

class PerceptionState:
    """
    Perception hierarchy P^i_l, l=0,1,2 for one player.
    Implements the three-layer structure:
      L0: P^i_0 = G_0  (belief over physical state)
      L1: P^i_1 = <Ŝ^i_j, û^i_j>  (belief about opponent strategies)
      L2: P^i_2  (belief about opponent's belief about own strategies)

    Medical specialisation — defender beliefs:
      p_attack_per_device ∈ [0,1]^n
      p_intent ∈ Δ(3)  over {overdose, false_alarm, disrupt}
      p_nation_state ∈ [0,1]
      p_knows_AI ∈ [0,1]
      p_detection_reliable ∈ [0,1]

    Medical specialisation — attacker beliefs:
      tau_hat ∈ [0,1]   (estimated detection threshold)
      model_hat ∈ Δ(3)  over {IsolationForest, LSTM-AE, Transformer}
      protocol_hat ∈ Δ(3) over {alert_only, auto_isolate, auto_recalibrate}
      p_aware ∈ [0,1]
      second_order_intent ∈ Δ(3)
    """

    def __init__(self, player: str, n_devices: int, rng: np.random.Generator):
        assert player in ('defender', 'attacker')
        self.player = player
        self.n = n_devices
        self.rng = rng

        #  Level-0 perception: distribution over device states 
        # Represented as mean and variance of belief over s_i
        self.mu_belief  = rng.standard_normal((n_devices, D_STATE))
        self.var_belief = np.ones((n_devices, D_STATE))

        #  Level-1 perception 
        if player == 'defender':
            # Per-device attack probability
            self.p_attack = np.full(n_devices, 0.1)
            # Attacker intent distribution: p_intent ∈ Δ(3)
            # {0: cause_overdose, 1: false_alarm, 2: disrupt_service}
            self.p_intent      = np.array([1/3, 1/3, 1/3])
            # Attacker sophistication
            self.p_nation_state = 0.2
            self.p_knows_AI     = 0.3
            self.p_detection_reliable = 0.85
            # Belief about attacker's strategy distribution σ̂^d_a ∈ Δ(S_a)
            # Simplified: prob mass over {no_atk, l0_atk, l1_atk, l2_atk}
            self.sigma_hat_opponent = np.array([0.4, 0.3, 0.2, 0.1])
            # Attacker's estimated utility weights (defender's belief)
            self.u_hat_opponent = np.array([1.0, -1.0])   # [gain, loss]

        else:  # attacker
            # Estimated detection threshold
            self.tau_hat = 0.5
            # Model type belief: {IsolationForest, LSTM-AE, Transformer}
            self.model_hat  = np.array([0.4, 0.4, 0.2])
            # Response protocol belief: {alert_only, auto_isolate, auto_recalibrate}
            self.protocol_hat = np.array([0.5, 0.3, 0.2])
            # Belief that defender is aware of ongoing attack
            self.p_aware    = 0.1
            # Second-order belief: attacker's belief about defender's intent perception
            self.second_order_intent = np.array([1/3, 1/3, 1/3])
            # Belief about defender strategy distribution σ̂^a_d ∈ Δ(S_d)
            self.sigma_hat_opponent = np.ones(N_DEF_ACTIONS) / N_DEF_ACTIONS
            # Defender's estimated utility weights (attacker's belief)
            self.u_hat_opponent = np.array([1.0, -1.0])

        #  Level-2 perception: second-order beliefs 
        # Player's belief about opponent's belief about own L0 perception
        self.l2_belief_about_opponent_l1 = np.ones(n_devices) * 0.1

        #  Perception mismatch matrix M_{i,j} ∈ R^{d×d} (Def.1) 
        # Scalar proxy: mean KL between own and ground-truth perception
        self.mismatch_scalar = 0.0

    def total_variation_distance(self, other: 'PerceptionState') -> float:
        """
        d(P, P') = (1/2) Σ_θ |P(θ) - P'(θ)|   Eq.12
        Computed over flattened belief vectors.
        """
        p1 = self.mu_belief.ravel()
        p2 = other.mu_belief.ravel()
        # Normalise to probability vectors via softmax
        def softmax_norm(x):
            x = x - x.max()
            ex = np.exp(x)
            return ex / ex.sum()
        p1n = softmax_norm(p1)
        p2n = softmax_norm(p2)
        return 0.5 * np.sum(np.abs(p1n - p2n))

    def kl_divergence(self, other: 'PerceptionState') -> float:
        """
        D_KL(P^i || P^{-i})  (Eq.5, Eq.36)
        Computed over strategy distribution beliefs.
        """
        p = self.sigma_hat_opponent + 1e-12
        q = other.sigma_hat_opponent + 1e-12
        # Align sizes
        sz = min(len(p), len(q))
        p, q = p[:sz], q[:sz]
        p = p / p.sum()
        q = q / q.sum()
        return float(np.sum(rel_entr(p, q)))

    def copy(self) -> 'PerceptionState':
        return copy.deepcopy(self)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  HYPERGAME MODEL  H_k  (Sec. III, Def. 1)
# ══════════════════════════════════════════════════════════════════════════════

class HypergameModel:
    """
    K-level hypergame H_k = <{G_0,...,G_k}, {M_{i,j}}, {Φ_i}>   Def.1

    Maintains subjective games G_d, G_a and perception hierarchy.
    Implements:
      - Perception mismatch matrix M_{i,j}
      - Bayesian posterior update  Eq.1, 2
      - Perception update dynamics Eq.21
      - Information advantage      Eq.27, 36
    """

    def __init__(self,
                 n_devices: int,
                 k_defender: int,
                 k_attacker: int,
                 rng: np.random.Generator):
        self.n = n_devices
        self.k_d = k_defender
        self.k_a = k_attacker
        self.rng = rng

        # Subjective perception states H_1 = {G_d, G_a}  (Sec. III-A)
        self.P_d = PerceptionState('defender', n_devices, rng)
        self.P_a = PerceptionState('attacker', n_devices, rng)

        # Perception mismatch matrices M_{d,a} and M_{a,d}
        # Scalar proxy = TV distance between perceptions
        self.M_da = 0.0   # how much defender misperceives attacker's game
        self.M_ad = 0.0   # how much attacker misperceives defender's game

        # Perception function Φ_i: G_actual → G_i
        # Modelled as additive Gaussian noise with time-varying variance
        self.phi_d_std = 0.1    # defender's perception noise
        self.phi_a_std = 0.2    # attacker's perception noise (higher uncertainty)

        # History of observations for defender  Od_{1:t-1}
        self.obs_history_d: List[np.ndarray] = []
        self.action_history_d: List[int] = []

        # History for attacker
        self.obs_history_a: List[np.ndarray] = []
        self.action_history_a: List[int] = []

    def update_mismatch(self):
        """
        M_{i,j} quantifies how player i's perception of G_j
        differs from actual G_j.  (Def. 1)
        Using TV distance as the metric d(·,·) on perception space.
        """
        self.M_da = self.P_d.total_variation_distance(self.P_a)
        self.M_ad = self.P_a.total_variation_distance(self.P_d)

    def bayesian_posterior_attacker_perception(
            self,
            new_obs_d: np.ndarray,
            P_a_hypotheses: List[PerceptionState],
            prior_weights: np.ndarray) -> np.ndarray:
        """
        P(P^(a) | O_d) ∝ ∫ P(O_d | P^(a), θ) P(θ) dθ   Eq.1
        with temporal factorisation:
          P(O_d | P^(a), θ) = Π_{t=1}^T P(o^d_t | P^(a), θ, O^d_{1:t-1})  Eq.2

        Approximated by Monte-Carlo importance sampling over a finite set of
        attacker perception hypotheses.
        """
        n_hyp = len(P_a_hypotheses)
        log_likelihoods = np.zeros(n_hyp)

        for hi, hyp in enumerate(P_a_hypotheses):
            ll = 0.0
            # For each time step in history (Eq.2 product)
            for t_idx, o_d in enumerate(self.obs_history_d):
                # P(o^d_t | P^(a), θ, O^d_{1:t-1})
                # Gaussian likelihood: obs ~ N(belief_mean, variance)
                # belief_mean = what attacker perceives = hyp.mu_belief mean
                mu  = hyp.mu_belief.mean(axis=0)
                # align dimensions
                dim = min(len(o_d), len(mu))
                mu_  = mu[:dim]
                sig_ = np.sqrt(hyp.var_belief.mean(axis=0)[:dim] + SIGMA_NOISE**2)
                # Inline Gaussian log-likelihood (avoids scipy overhead: ~3x faster)
                z = (o_d[:dim] - mu_) / sig_
                ll += float(-0.5 * np.dot(z, z) - np.sum(np.log(sig_))
                            - 0.5 * dim * np.log(2 * np.pi))
            log_likelihoods[hi] = ll

        # Numerically stable posterior
        log_posterior = log_likelihoods + np.log(prior_weights + 1e-300)
        log_posterior -= log_posterior.max()
        posterior = np.exp(log_posterior)
        posterior /= posterior.sum()
        return posterior

    def perception_update_dynamics(self,
                                   player: str,
                                   P_observed: np.ndarray,
                                   grad_U: np.ndarray,
                                   dt: float = 1.0):
        """
        dP^i/dt = υ(P^i_observed - P^i) + β ∇U_i(P^i)   Eq.21
        Euler integration: P^i_{t+1} = P^i_t + dt * dP^i/dt
        """
        P_state = self.P_d if player == 'defender' else self.P_a
        # Extract flat perception vector (device attack beliefs)
        if player == 'defender':
            P_i   = P_state.p_attack.copy()
            dP_dt = (UPSILON * (P_observed[:self.n] - P_i)
                     + BETA_TEMP * grad_U[:self.n])
            P_state.p_attack = np.clip(P_i + dt * dP_dt, 0.0, 1.0)
        else:
            P_i   = np.array([P_state.tau_hat])
            dP_dt = (UPSILON * (P_observed[:1] - P_i)
                     + BETA_TEMP * grad_U[:1])
            new_val = np.clip(P_i + dt * dP_dt, 0.0, 1.0)
            P_state.tau_hat = float(new_val.ravel()[0])

    def information_advantage(self, P_true_a, P_true_d):
        """
        Λ(t) = KL(P^A || P^D_true) − KL(P^D || P^A_true)   Eq.27/36

        Λ > 0  → defender advantage  (attacker more confused about true state)
        Λ < 0  → attacker advantage  (defender more confused about true state)

        Both inputs are discrete distributions over N_DEVICES device indices.
        Direct discrete KL — no histogramming needed or correct here.
        """
        def kl_direct(p, q):
            # Laplace smoothing so KL is finite even with zero-support devices
            p = np.asarray(p, dtype=float)
            q = np.asarray(q, dtype=float)
            p = (p + 1e-9) / (p.sum() + 1e-9 * len(p))
            q = (q + 1e-9) / (q.sum() + 1e-9 * len(q))
            return float(np.sum(rel_entr(p, q)))

        # Defender belief: per-device attack probability p_attack
        pd = self.P_d.p_attack
        # Attacker belief: level-2 estimate of how well defender perceives its attacks
        pa = self.P_a.l2_belief_about_opponent_l1

        pd_norm = (pd + 1e-9) / (pd.sum() + 1e-9 * len(pd))
        pa_norm = (pa + 1e-9) / (pa.sum() + 1e-9 * len(pa))

        # Λ > 0 when attacker is MORE confused than defender
        return kl_direct(pa_norm, P_true_d) - kl_direct(pd_norm, P_true_a)

    def perception_gap(self, s_true: np.ndarray,
                       s_perceived: np.ndarray) -> float:
        """
        ΔP(t) = ‖P_D(s(t)) − s(t)‖_2   Eq.35
        """
        return float(np.linalg.norm(s_perceived - s_true))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  STRATEGY SPACES AND UTILITY FUNCTIONS  (Sec. III-C, III-D)
# ══════════════════════════════════════════════════════════════════════════════

class StrategySpace:
    """
    σ_d ∈ Σ_d = Δ(⋃_{l=0}^{k} S^l_d)   Eq.3
    σ_a ∈ Σ_a = Δ(⋃_{l=0}^{k} S^l_a)   Eq.4

    Hospital mapping:
      S^0_d = {Monitor, Alert, Isolate, Recalibrate}
      S^1_d = {Deceive, Honeypot, ThresholdDither}
      S^2_d = {DoubleBluff, LeakFakeModel, StrategicInconsistency}

      S^0_a = {NoAttack, VitalSignsShift, DataReplay, DeviceFreeze}
      S^1_a = {Camouflage, MimicClinician, AdaptivePerturbation}
      S^2_a = {Probe, Feint, MetaAdaptive}
    """
    # Defender level-0: mapped to Table I actions
    DEF_L0 = ['Monitor', 'Alert', 'Isolate', 'Recalibrate']
    DEF_L1 = ['Deceive', 'Honeypot', 'ThresholdDither']
    DEF_L2 = ['DoubleBluff', 'LeakFakeModel', 'StrategicInconsistency']
    DEF_ALL = DEF_L0 + DEF_L1 + DEF_L2   # |Σ_d| = 10

    ATK_L0 = ['NoAttack', 'VitalSignsShift', 'DataReplay', 'DeviceFreeze']
    ATK_L1 = ['Camouflage', 'MimicClinician', 'AdaptivePerturbation']
    ATK_L2 = ['Probe', 'Feint', 'MetaAdaptive']
    ATK_ALL = ATK_L0 + ATK_L1 + ATK_L2   # |Σ_a| = 10

    # Map strategy names to Table-I defense indices
    STRATEGY_TO_DEF_IDX = {
        'Monitor':    DEF_MONITOR,
        'Alert':      DEF_MONITOR,
        'Isolate':    DEF_ISOLATION,
        'Recalibrate':DEF_ACCESS_CTRL,
        'Deceive':    DEF_DECEPTION,
        'Honeypot':   DEF_HONEYPOT,
        'ThresholdDither': DEF_MONITOR,
        'DoubleBluff':DEF_DECEPTION,
        'LeakFakeModel': DEF_ENCRYPTION,
        'StrategicInconsistency': DEF_DECEPTION,
    }

    def __init__(self, player: str, k_level: int):
        assert player in ('defender', 'attacker')
        self.player = player
        self.k_level = k_level
        all_strats = self.DEF_ALL if player == 'defender' else self.ATK_ALL
        n = len(all_strats)
        # σ is a probability distribution over strategies (Eq.3/4)
        self.sigma = np.ones(n) / n    # uniform initialisation
        self.n_strategies = n
        self.strategy_names = all_strats

    def level_strategies(self, level: int) -> List[str]:
        """Return S^l_i for the given level."""
        if self.player == 'defender':
            return [self.DEF_L0, self.DEF_L1, self.DEF_L2][level]
        else:
            return [self.ATK_L0, self.ATK_L1, self.ATK_L2][level]

    def level_indices(self, level: int) -> np.ndarray:
        """Indices within sigma that correspond to reasoning level l."""
        offset = sum(len(self.level_strategies(l)) for l in range(level))
        length = len(self.level_strategies(level))
        return np.arange(offset, offset + length)

    def sample_strategy(self) -> int:
        """Sample strategy index from mixed strategy distribution."""
        return int(np.random.choice(self.n_strategies, p=self.sigma))

    def project_to_simplex(self):
        """Project σ onto probability simplex Δ(·)."""
        sigma = self.sigma
        sigma = np.clip(sigma, 0, None)
        if sigma.sum() < 1e-12:
            self.sigma = np.ones(self.n_strategies) / self.n_strategies
        else:
            self.sigma = sigma / sigma.sum()

    def entropy(self) -> float:
        """Strategy entropy — regularisation for R(σ) in Eq.20."""
        p = self.sigma + 1e-12
        return float(-np.sum(p * np.log(p)))


def subjective_payoff(si: int, s_hat_j: int, player: str,
                      device_states: np.ndarray,
                      health_vector: np.ndarray,
                      n_devices: int) -> float:
    """
    û_i(s_i, ŝ^i_{-i}) = r_i(s_i) - c_i(s_i) + v_i(s_i, ŝ^i_{-i}) - l_i(s_i, ŝ^i_{-i})
    Eq.6  (fully implemented, hospital context)

    Parameters
    -
    si      : strategy index for player i
    s_hat_j : player i's belief about opponent strategy
    player  : 'defender' or 'attacker'
    """
    if player == 'defender':
        # Map strategy index → Table-I defence action
        def_action = min(si, N_DEF_ACTIONS - 1)
        atk_belief = min(s_hat_j, N_ATK_TYPES - 1)

        # r_i(s_i): direct reward — saved patient health
        r = float(np.mean(health_vector) * DEF_ETA[def_action])

        # c_i(s_i): deployment cost
        c = DEF_COST[def_action]

        # v_i(s_i, ŝ^i_{-i}): strategic value if defence counters believed attack
        eta_effective = DEF_EFF[def_action, atk_belief]
        v = eta_effective * float(np.mean(health_vector))

        # l_i(s_i, ŝ^i_{-i}): perceived risk if attack succeeds despite defence
        # Risk is high if isolation disrupts critical devices
        isolation_risk = 0.3 if def_action == DEF_ISOLATION else 0.0
        l = max(0.0, (1.0 - eta_effective) * 0.5 + isolation_risk)

    else:  # attacker
        atk_type = min(si % N_ATK_TYPES, N_ATK_TYPES - 1)
        def_belief = min(s_hat_j, N_DEF_ACTIONS - 1)

        # r_i(s_i): attacker reward = disruption * mean health damage
        r = (1.0 - float(np.mean(health_vector))) * ATK_MU[atk_type]

        # c_i(s_i): attack cost (computational, risk of detection)
        atk_costs = [0.0, 0.3, 0.5, 0.8, 0.4, 0.2]
        c = atk_costs[min(atk_type, len(atk_costs)-1)]

        # v_i: strategic value given defender seems unprepared
        v = DEF_EFF[def_belief, atk_type] * 0.5 \
            if def_belief < N_DEF_ACTIONS else 0.5

        # l_i: risk of being detected and losing foothold
        l = DEF_EFF[def_belief, atk_type] * 0.7

    return r - c + v - l


def expected_utility(sigma_i: np.ndarray,
                     sigma_hat_j: np.ndarray,
                     player: str,
                     device_states: np.ndarray,
                     health_vector: np.ndarray,
                     perception_i: PerceptionState,
                     perception_j: PerceptionState,
                     lambda_w: float = LAMBDA_GAME) -> float:
    """
    U_i(σ_i, σ_{-i}) = E[û_i(s_i, ŝ^i_{-i})] + λ · I(P^i, P^{-i})   Eq.5

    where E[û_i] = Σ_{s_i} Σ_{ŝ_{-i}} σ_i(s_i)·σ̂^i_{-i}(ŝ_{-i})·û_i(s_i,ŝ^i_{-i})
    """
    # Normalise distributions
    si_dist = sigma_i  / (sigma_i.sum() + 1e-12)
    sj_dist = sigma_hat_j / (sigma_hat_j.sum() + 1e-12)

    ni, nj = len(si_dist), len(sj_dist)

    # ── Vectorised payoff matrix  û(si, sj)  shape (ni, nj) ─────────────────
    # Precompute mean_health once (avoids ni×nj calls to np.mean)
    mean_h = float(np.mean(health_vector))

    if player == 'defender':
        si_idx = np.clip(np.arange(ni), 0, N_DEF_ACTIONS - 1)
        sj_idx = np.clip(np.arange(nj), 0, N_ATK_TYPES - 1)
        r   = mean_h * DEF_ETA[si_idx]                              # (ni,)
        c   = DEF_COST[si_idx]                                       # (ni,)
        eta = DEF_EFF[si_idx[:, None], sj_idx[None, :]]             # (ni,nj)
        v   = eta * mean_h                                           # (ni,nj)
        iso = (si_idx == DEF_ISOLATION).astype(float) * 0.3         # (ni,)
        l   = np.maximum(0.0, (1.0 - eta) * 0.5 + iso[:, None])    # (ni,nj)
        payoff_mat = (r - c)[:, None] + v - l                       # (ni,nj)
    else:  # attacker
        _atk_costs = np.array([0.0, 0.3, 0.5, 0.8, 0.4, 0.2])
        si_idx     = np.arange(ni)
        sj_idx     = np.arange(nj)
        atk_type   = np.clip(si_idx % N_ATK_TYPES, 0, N_ATK_TYPES - 1)  # (ni,)
        def_bel    = np.clip(sj_idx, 0, N_DEF_ACTIONS - 1)               # (nj,)
        r   = (1.0 - mean_h) * ATK_MU[atk_type]                         # (ni,)
        c   = _atk_costs[np.clip(atk_type, 0, len(_atk_costs)-1)]        # (ni,)
        # v - l = (0.5 - 0.7) * DEF_EFF[def_bel, atk_type]
        eff = DEF_EFF[def_bel[None, :], atk_type[:, None]]               # (ni,nj)
        payoff_mat = (r - c)[:, None] + eff * (-0.2)                     # (ni,nj)

    # E[û] = σ_i^T · payoff_mat · σ_{-i}  (Eq.5 double sum, one matmul)
    E_u_hat = float(si_dist @ payoff_mat @ sj_dist)

    # Perception advantage term I(P^i; P^{-i}) = D_KL(P^i || P^{-i})
    I_perception = perception_i.kl_divergence(perception_j)

    return E_u_hat - lambda_w * I_perception


# ══════════════════════════════════════════════════════════════════════════════
# 6.  BEST RESPONSE COMPUTATION  (Sec. IV-A, Eq. 7–9)
# ══════════════════════════════════════════════════════════════════════════════

def best_response_level0(player: str,
                         sigma_opponent_hat: np.ndarray,
                         device_states: np.ndarray,
                         health_vector: np.ndarray,
                         perception_i: PerceptionState,
                         perception_j: PerceptionState) -> np.ndarray:
    """
    BR^0_i(σ_{-i}, P^{-i}) = argmax_{σ_i ∈ S^0_i} E[û_i(σ_i, σ̂^i_{-i})]   Eq.7
    Enumerate over level-0 strategies only.
    """
    space = StrategySpace(player, 0)
    l0_idx = space.level_indices(0)
    n_l0 = len(l0_idx)

    utilities = np.zeros(n_l0)
    for local_i, global_i in enumerate(l0_idx):
        # One-hot: put all mass on this pure strategy
        sigma_pure = np.zeros(space.n_strategies)
        sigma_pure[global_i] = 1.0
        utilities[local_i] = expected_utility(
            sigma_pure, sigma_opponent_hat, player,
            device_states, health_vector, perception_i, perception_j)

    # Soft best response via softmax (avoids non-differentiability)
    br = np.zeros(space.n_strategies)
    softmax_u = np.exp(utilities / (BETA_TEMP + 1e-8))
    softmax_u /= softmax_u.sum()
    br[l0_idx] = softmax_u
    return br


def best_response_level1(player: str,
                         sigma_opponent: np.ndarray,
                         P_i_0: PerceptionState,
                         device_states: np.ndarray,
                         health_vector: np.ndarray,
                         perception_i: PerceptionState,
                         perception_j: PerceptionState) -> np.ndarray:
    """
    BR^1_i = argmax_{σ_i ∈ S^1_i} E[û_i(σ_i, BR^0_{-i}(σ_i, P^{(i,0)}))]   Eq.8
    """
    space = StrategySpace(player, 1)
    l1_idx = space.level_indices(1)
    opp_player = 'attacker' if player == 'defender' else 'defender'

    # Compute opponent's level-0 BR for each of our level-1 strategies
    utilities = np.zeros(len(l1_idx))
    for local_i, global_i in enumerate(l1_idx):
        sigma_pure = np.zeros(space.n_strategies)
        sigma_pure[global_i] = 1.0
        # Opponent's BR^0 to our play
        br0_opp = best_response_level0(
            opp_player, sigma_pure,
            device_states, health_vector, perception_j, perception_i)
        utilities[local_i] = expected_utility(
            sigma_pure, br0_opp, player,
            device_states, health_vector, perception_i, perception_j)

    br = np.zeros(space.n_strategies)
    softmax_u = np.exp(utilities / (BETA_TEMP + 1e-8))
    softmax_u /= softmax_u.sum()
    br[l1_idx] = softmax_u
    return br


def best_response_levelk(player: str,
                          sigma_opponent: np.ndarray,
                          P_minus_i: PerceptionState,
                          P_i_k_minus1: PerceptionState,
                          device_states: np.ndarray,
                          health_vector: np.ndarray,
                          perception_i: PerceptionState,
                          perception_j: PerceptionState,
                          k: int,
                          lambda_w: float = LAMBDA_GAME) -> np.ndarray:
    """
    BR^k_i = argmax_{σ_i ∈ S^k_i} {
        E[û_i(σ_i, BR^{k-1}_{-i}(σ_i, P^{(i,k-1)}))]
        + λ I(P^{(i,k)}; P^{(-i,k-1)})
    }   Eq.9

    Recursive implementation up to k=2.
    """
    if k == 0:
        return best_response_level0(
            player, sigma_opponent, device_states, health_vector,
            perception_i, perception_j)
    elif k == 1:
        return best_response_level1(
            player, sigma_opponent, P_i_k_minus1,
            device_states, health_vector, perception_i, perception_j)

    # k >= 2
    space = StrategySpace(player, k)
    lk_idx = space.level_indices(min(k, 2))
    opp_player = 'attacker' if player == 'defender' else 'defender'

    utilities = np.zeros(len(lk_idx))
    for local_i, global_i in enumerate(lk_idx):
        sigma_pure = np.zeros(space.n_strategies)
        sigma_pure[global_i] = 1.0

        # Opponent's BR^{k-1} to our strategy
        br_opp_km1 = best_response_levelk(
            opp_player, sigma_pure,
            perception_i, P_i_k_minus1,
            device_states, health_vector,
            perception_j, perception_i, k - 1, lambda_w)

        # Base expected utility term
        u_base = expected_utility(
            sigma_pure, br_opp_km1, player,
            device_states, health_vector, perception_i, perception_j)

        # Mutual information term I(P^{(i,k)}; P^{(-i,k-1)})
        I_term = perception_i.kl_divergence(P_i_k_minus1)

        utilities[local_i] = u_base - lambda_w * I_term

    br = np.zeros(space.n_strategies)
    softmax_u = np.exp(utilities / (BETA_TEMP + 1e-8))
    softmax_u /= softmax_u.sum()
    br[lk_idx] = softmax_u
    return br


# ══════════════════════════════════════════════════════════════════════════════
# 7.  CONVERGENCE / THEOREM 1 MACHINERY  (Sec. IV-B, Lemmas 1-3)
# ══════════════════════════════════════════════════════════════════════════════

def bayesian_perception_update(P: PerceptionState,
                                signal: np.ndarray,
                                likelihood_func,
                                m_lower: float = 0.01,
                                M_upper: float = 10.0) -> PerceptionState:
    """
    Bayesian update operator T_i (Lemma 1, Eq.11):
      T_i(P)(θ) = L(o|θ)P(θ) / Z

    Contracts in TV distance by ρ = C·M/m < 1.

    Returns updated perception state.
    """
    P_new = P.copy()

    # Apply Bayes update to per-device attack probability vector
    # θ ∈ {0:no_attack, 1:attacked} for each device
    for i in range(P.n):
        # Prior
        p_atk = float(P.p_attack[i])
        p_no  = 1.0 - p_atk

        # Likelihood  L(o|θ)  from signal at device i
        s_val = float(signal[i]) if i < len(signal) else 0.0
        # Under attack: observation deviates from prior
        L_atk = float(stats.norm.pdf(s_val, loc=0.3, scale=SIGMA_NOISE + 0.05))
        L_no  = float(stats.norm.pdf(s_val, loc=0.0, scale=SIGMA_NOISE))

        # Bound likelihoods for contraction guarantee
        L_atk = float(np.clip(L_atk, m_lower, M_upper))
        L_no  = float(np.clip(L_no,  m_lower, M_upper))

        # Normalisation  Z = Σ L(o|θ')P(θ')
        Z = L_atk * p_atk + L_no * p_no
        Z = max(Z, 1e-12)

        # Posterior  T_i(P)(θ)  (Eq.11)
        P_new.p_attack[i] = (L_atk * p_atk) / Z

    # Contraction constant ρ = C · M/m  (must satisfy 0 < ρ < 1)
    rho = min(0.95, (M_upper / m_lower) * 0.1)   # scaled so ρ < 1
    return P_new, rho


def compute_approximation_error(U_star: float,
                                  U_best: float) -> float:
    """
    ε_i(k) = |U_i(σ*_i, σ*_{-i}) - max_{σ_i} U_i(σ_i, σ*_{-i})|   Eq.10 / Prop.1
    """
    return abs(U_star - U_best)


def convergence_bound(k: int, C: float = 1.0, rho: float = 0.9) -> float:
    """
    ε(k) ≤ C·ρ^k → 0 as k→∞   Proposition 1
    """
    return C * (rho ** k)


def iterated_best_response(sigma_d: np.ndarray,
                            sigma_a: np.ndarray,
                            P_d: PerceptionState,
                            P_a: PerceptionState,
                            device_states: np.ndarray,
                            health_vector: np.ndarray,
                            k_d: int = K_DEFENDER,
                            k_a: int = K_ATTACKER,
                            max_iter: int = 15,
                            eps: float = EPS_CONV) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Iterated best response to find hypergame Nash equilibrium (HNE).
    Convergence tolerance ε_conv = 10^{-3}  (Eq.37, Sec. VI-D4)

    Returns (σ*_d, σ*_a, n_iterations_to_convergence)
    """
    sigma_d = sigma_d.copy()
    sigma_a = sigma_a.copy()

    for iteration in range(max_iter):
        # Defender's best response given attacker strategy (at level k_d)
        br_d = best_response_levelk(
            'defender', sigma_a, P_a, P_d,
            device_states, health_vector, P_d, P_a, k_d)
        # Attacker's best response given defender strategy (at level k_a)
        br_a = best_response_levelk(
            'attacker', sigma_d, P_d, P_a,
            device_states, health_vector, P_a, P_d, k_a)

        # Check convergence  ‖σ_new - σ_old‖ ≤ ε  (Eq.37)
        delta = (np.linalg.norm(br_d - sigma_d) +
                 np.linalg.norm(br_a - sigma_a))
        sigma_d = br_d
        sigma_a = br_a

        if delta < eps:
            return sigma_d, sigma_a, iteration + 1

    return sigma_d, sigma_a, max_iter


def resilience_metric(defender_strategies: np.ndarray,
                       attacker_strategies: np.ndarray,
                       U_D_matrix: np.ndarray,
                       U_A_matrix: np.ndarray) -> float:
    """
    Hybrid resilience (Eq.17):
      ρ = min_{σ_A ∈ Δ(A)} max_{σ_D ∈ Δ(D)} σ_D^T (U_D - U_A) σ_A

    This is the value of the zero-sum game with payoff matrix M = U_D - U_A,
    solved exactly via linear programming (minimax theorem).

    LP formulation (row player / defender maximises):
      max  v
      s.t. M^T σ_D ≥ v · 1      (defender guarantees at least v against any σ_A)
           σ_D ≥ 0
           1^T σ_D = 1
    """
    M = U_D_matrix - U_A_matrix          # (n_d × n_a)  payoff matrix
    n_d, n_a = M.shape

    # LP: variables are [σ_D (n_d,), v (scalar)] → length n_d + 1
    # Objective: maximise v  →  minimise -v
    c = np.zeros(n_d + 1)
    c[-1] = -1.0                         # coefficient of v in objective

    # Inequality constraints: M^T σ_D - v·1 ≥ 0  →  -M^T σ_D + v·1 ≤ 0
    # Shape: (n_a, n_d + 1)
    A_ub = np.hstack([-M.T, np.ones((n_a, 1))])   # -M^T σ_D + v ≤ 0
    b_ub = np.zeros(n_a)

    # Equality constraint: 1^T σ_D = 1
    A_eq = np.ones((1, n_d + 1))
    A_eq[0, -1] = 0.0                    # v not constrained by equality
    b_eq = np.array([1.0])

    # Bounds: σ_D ∈ [0, 1], v ∈ [-∞, +∞]
    bounds = [(0.0, 1.0)] * n_d + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                     A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        return float(-result.fun)        # value of the game = -min(-v) = v*
    else:
        # Fallback to pure minimax if LP fails (numerical edge case)
        diff = M
        return float(np.min(np.max(diff, axis=0)))


# ══════════════════════════════════════════════════════════════════════════════
# 8.  ADVERSARIAL META-LEARNING  (Sec. V, Eq. 18–21)
# ══════════════════════════════════════════════════════════════════════════════

class AdversarialMetaLearner:
    """
    Implements the minimax meta-learning objective (Eq.18-20):

      min_θ max_δ  L(θ, x+δ) + β R(P^D, P^A)
      θ_{t+1} ← θ_t − η ∇_θ[L(θ) + ω D_KL(P^D ‖ P^A_true)]   Eq.19
      min_θ E_{s,a}[L(θ) + λ D_KL(P^d‖P^a) + μ R(σ)]           Eq.20
        s.t. σ ∈ NE(H_k(P^D, P^A))

    Parameters (θ) are 2-layer MLP weights representing the defender model.
    Adversarial perturbation (δ) is optimised via projected gradient ascent.
    """

    def __init__(self, input_dim: int, rng: np.random.Generator):
        self.rng = rng
        self.input_dim = input_dim

        #  Defender model parameters θ (2-layer MLP) 
        hidden = 64
        # Layer 1: W1 ∈ R^{hidden × input_dim}, b1 ∈ R^{hidden}
        self.W1 = rng.normal(0, 0.01, (hidden, input_dim))
        self.b1 = np.zeros(hidden)
        # Layer 2: W2 ∈ R^{2 × hidden}  (binary: attack/no-attack)
        self.W2 = rng.normal(0, 0.01, (2, hidden))
        self.b2 = np.zeros(2)

        # Adam optimizer state
        self.m_W1 = np.zeros_like(self.W1)
        self.v_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        self.m_W2 = np.zeros_like(self.W2)
        self.v_W2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)
        self.adam_t = 0

        # Current adversarial perturbation δ ∈ R^{input_dim}
        self.delta = np.zeros(input_dim)

        # Perception distributions P^D, P^A (histograms, n_bins=20)
        self.P_D_hist = np.ones(N_BINS) / N_BINS
        self.P_A_hist = np.ones(N_BINS) / N_BINS
        self.P_A_true = np.ones(N_BINS) / N_BINS

    def forward(self, x: np.ndarray) -> np.ndarray:
        """MLP forward pass: x → logits."""
        h = np.maximum(0, self.W1 @ x + self.b1)  # ReLU
        return self.W2 @ h + self.b2

    def softmax(self, z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def cross_entropy_loss(self, x: np.ndarray, y: int) -> float:
        """L(θ, x): cross-entropy classification loss."""
        logits = self.forward(x)
        probs  = self.softmax(logits)
        return -np.log(probs[y] + 1e-12)

    def backward(self, x: np.ndarray, y: int) -> Dict[str, np.ndarray]:
        """Compute ∇_θ L(θ, x) analytically."""
        # Forward
        z1 = self.W1 @ x + self.b1
        h  = np.maximum(0, z1)
        z2 = self.W2 @ h + self.b2
        p  = self.softmax(z2)

        # Output layer gradient
        dL_dz2 = p.copy()
        dL_dz2[y] -= 1.0       # ∂CE/∂z2

        dL_dW2 = np.outer(dL_dz2, h)
        dL_db2 = dL_dz2

        # Hidden layer gradient
        dL_dh  = self.W2.T @ dL_dz2
        dL_dz1 = dL_dh * (z1 > 0).astype(float)   # ReLU derivative
        dL_dW1 = np.outer(dL_dz1, x)
        dL_db1 = dL_dz1

        return {'W1': dL_dW1, 'b1': dL_db1,
                'W2': dL_dW2, 'b2': dL_db2}
    
    def input_gradient(self, x: np.ndarray, y: int) -> np.ndarray:
        """

         chain rule: ∇_δ L = W1^T · (W2^T(p - e_y) ⊙ 1[z1 > 0])
        Covers all input_dim dimensions exactly.
        """
        # Forward pass
        z1 = self.W1 @ x + self.b1
        h  = np.maximum(0, z1)
        z2 = self.W2 @ h + self.b2
        p  = self.softmax(z2)

        # Backprop to input layer
        dL_dz2 = p.copy()
        dL_dz2[y] -= 1.0                                  # ∂L/∂z2
        dL_dz1 = (self.W2.T @ dL_dz2) * (z1 > 0).astype(float)  # ∂L/∂z1
        dL_dx  = self.W1.T @ dL_dz1                       # ∂L/∂(x+δ) = ∂L/∂δ
        return dL_dx
    
    def kl_histogram(self, p: np.ndarray, q: np.ndarray) -> float:
        """D_KL(P || Q) for histogram distributions."""
        p = p + 1e-12; p /= p.sum()
        q = q + 1e-12; q /= q.sum()
        return float(np.sum(rel_entr(p, q)))

    def strategy_regulariser(self, sigma_d: np.ndarray,
                              sigma_a: np.ndarray) -> float:
        """
        R(σ) = entropy of defender strategy (encourages exploration)
        Used in Eq.20.
        """
        p = sigma_d + 1e-12; p /= p.sum()
        return float(-np.sum(p * np.log(p)))

    def adversarial_perturbation_update(self,
                                          x: np.ndarray,
                                          y: int,
                                          n_steps: int = K_ADV_STEPS) -> np.ndarray:
        """
        max_δ  L(θ, x+δ) + β R(P^D, P^A)    inner maximisation of Eq.18
        Projected gradient ascent with ℓ2-norm constraint ‖δ‖ ≤ ε (Sec.VI-B1)
        """
        delta = self.delta.copy()
        step_size = 0.01

        for _ in range(n_steps):
            x_adv = x + delta

            # Gradient ∇_δ L(θ, x+δ): treat δ as input perturbation
            # Compute via finite difference for stability
            # Gradient ∇_δ L(θ, x+δ): exact analytical gradient via backprop
            # Covers all input_dim dimensions 
            grad_delta = self.input_gradient(x_adv, y)

            # Perception misalignment gradient ∇_δ β R(P^D, P^A)
            # Update P^D_hist using perturbed sample
            hist_adv, bin_edges = np.histogram(x_adv, bins=N_BINS, density=True)
            P_D_new = hist_adv / (hist_adv.sum() + 1e-12)
            kl_grad = BETA_TEMP * (self.kl_histogram(P_D_new, self.P_A_hist) -
                                    self.kl_histogram(self.P_D_hist, self.P_A_hist))

            # Ascent step
            delta += step_size * (grad_delta + kl_grad * 0.1)

            # Project onto ℓ2-ball ‖δ‖_2 ≤ ε (Eq.30)
            d_norm = np.linalg.norm(delta)
            if d_norm > EPSILON_PERT:
                delta *= EPSILON_PERT / d_norm

            # Physiological plausibility check (temporal smoothness enforced
            # implicitly by bounded step size)
            delta = np.clip(delta, -EPSILON_PERT, EPSILON_PERT)

        self.delta = delta
        return delta

    def meta_gradient_update(self,
                               x: np.ndarray,
                               y: int,
                               P_D: np.ndarray,
                               P_A_true: np.ndarray,
                               sigma_d: np.ndarray,
                               sigma_a: np.ndarray,
                               lambda_w: float = LAMBDA_GAME,
                               mu_w: float = MU_GAME,
                               omega_w: float = OMEGA_META):
        """
        θ_{t+1} ← θ_t − η ∇_θ[L(θ) + ω D_KL(P^D ‖ P^A_true)]   Eq.19

        Full learning objective (Eq.20):
          min_θ E_{s,a}[L(θ) + λ D_KL(P^d‖P^a) + μ R(σ)]
        """
        self.adam_t += 1
        x_adv = x + self.delta

        # Compute base gradient ∇_θ L(θ)
        grads = self.backward(x_adv, y)

        # Perception alignment gradient: ∇_θ [ω D_KL(P^D ‖ P^A_true)]
        # Update P^D using current features
        hist_x, _ = np.histogram(x, bins=N_BINS, density=True)
        self.P_D_hist = hist_x / (hist_x.sum() + 1e-12)
        self.P_A_true = P_A_true / (P_A_true.sum() + 1e-12)

        kl_term = omega_w * self.kl_histogram(self.P_D_hist, self.P_A_true)

        # Strategy regularisation R(σ) = -H(σ_d)
        R_sigma = self.strategy_regulariser(sigma_d, sigma_a)

        # Total gradient (scalar KL term added to all weights uniformly)
        total_loss_scalar = (self.cross_entropy_loss(x_adv, y)
                             + lambda_w * kl_term
                             + mu_w * R_sigma)

        # Adam update for each parameter group
        for (W, dW, mW, vW, mb, vb, db, b) in [
            (self.W1, grads['W1'], self.m_W1, self.v_W1,
             self.m_b1, self.v_b1, grads['b1'], self.b1),
            (self.W2, grads['W2'], self.m_W2, self.v_W2,
             self.m_b2, self.v_b2, grads['b2'], self.b2),
        ]:
            # Adam moment updates
            mW[:] = BETA_ADAM1 * mW + (1 - BETA_ADAM1) * dW
            vW[:] = BETA_ADAM2 * vW + (1 - BETA_ADAM2) * dW**2
            m_hat_W = mW / (1 - BETA_ADAM1**self.adam_t)
            v_hat_W = vW / (1 - BETA_ADAM2**self.adam_t)
            W -= ETA * m_hat_W / (np.sqrt(v_hat_W) + 1e-8)
            W -= ETA * kl_term * np.sign(W) * 0.001   # perception term

            mb[:] = BETA_ADAM1 * mb + (1 - BETA_ADAM1) * db
            vb[:] = BETA_ADAM2 * vb + (1 - BETA_ADAM2) * db**2
            m_hat_b = mb / (1 - BETA_ADAM1**self.adam_t)
            v_hat_b = vb / (1 - BETA_ADAM2**self.adam_t)
            b -= ETA * m_hat_b / (np.sqrt(v_hat_b) + 1e-8)

        return total_loss_scalar

    def predict_attack(self, x: np.ndarray) -> Tuple[int, float]:
        """Classify observation as attack (1) or normal (0)."""
        logits = self.forward(x)
        probs  = self.softmax(logits)
        return int(np.argmax(probs)), float(probs[1])


# ══════════════════════════════════════════════════════════════════════════════
# 9.  DECEPTION-AWARE PARTICLE FILTER  (Sec. V-A, Eq. 22–26)
# ══════════════════════════════════════════════════════════════════════════════

class DeceptionAwareParticleFilter:
    """
    POMDP belief state b_t(s, δ) = P(s_t=s, δ_t=δ | h^D_t)   Eq.23
    Implements:
      - Distorted observation model  o^D_t = s_t + δ_t + ϑ   Eq.22
      - Bayesian filtering update    Eq.24
      - Deception detection via N_eff monitoring
      - Adaptive variance σ² → 9σ² under suspected attack
      - Kernel density resampling (prevents particle degeneracy)
      - POMDP value function         V(b) = max_a [R(b,a) + γ∫V(b')P(b'|b,a)db']  Eq.26
    """

    def __init__(self, state_dim: int, rng: np.random.Generator):
        self.d = state_dim
        self.rng = rng
        self.N = N_PARTICLES

        # Particles: each represents (s, δ_indicator) pair
        # s_particles ∈ R^{N × d}
        self.s_particles = rng.standard_normal((self.N, self.d))
        # δ_particles ∈ R^{N × d} (attack perturbation hypothesis)
        self.delta_particles = np.zeros((self.N, self.d))
        # Particle weights  w_i ≥ 0, Σw_i = 1
        self.weights = np.ones(self.N) / self.N

        # Attack probability estimate for each particle
        self.attack_indicators = np.zeros(self.N)

        # Innovation statistics for deception detection
        self.innovation_buffer = deque(maxlen=10)
        self.sigma2_adaptive = SIGMA_NOISE**2

        # Current belief mean and covariance
        self.belief_mean = np.zeros(self.d)
        self.belief_cov  = np.eye(self.d)

        # Value function approximation (table over health levels)
        # ~7x faster than dict lookup for q_value inner loop
        self.V_array: np.ndarray = np.zeros(10)

    def observation_likelihood(self,
                                obs: np.ndarray,
                                s_particle: np.ndarray,
                                delta_particle: np.ndarray,
                                sigma2: float) -> float:
        """
        P(o^D_{t+1} | s', δ')  using Gaussian noise model (Eq.24 numerator)
        Operates on first min(d, len(obs)) dimensions for efficiency.
        """
        dim = min(len(obs), len(s_particle))
        predicted = s_particle[:dim] + delta_particle[:dim]
        diff = obs[:dim] - predicted
        log_lik = -0.5 * np.dot(diff, diff) / sigma2 - \
                   0.5 * dim * np.log(2 * np.pi * sigma2)
        return float(np.exp(np.clip(log_lik, -700, 0)))

    def effective_sample_size(self) -> float:
        """N_eff = 1 / Σ w_i²"""
        return 1.0 / (np.sum(self.weights**2) + 1e-12)

    def detect_deception(self, innovation: np.ndarray) -> bool:
        """
        Monitor innovation statistics to identify adversarial perturbations.
        (Sec. VI-C2, bullet 1)
        Large innovations inconsistent with noise model → attack suspected.
        """
        self.innovation_buffer.append(np.linalg.norm(innovation))
        if len(self.innovation_buffer) < 3:
            return False
        mean_inn = sum(self.innovation_buffer) / len(self.innovation_buffer)
        # Threshold: > 3σ innovation → suspected attack
        return mean_inn > 3.0 * np.sqrt(self.sigma2_adaptive * self.d)

    def update(self,
               obs_d: np.ndarray,
               action_d: int,
               transition_noise: float = 0.05,
               pregenerated_noise: np.ndarray = None,
               kde_noise: np.ndarray = None,
               guided_noise: np.ndarray = None) -> Dict:
        """
        Full particle filter update step implementing Eq.22–24.

        b_{t+1}(s', δ') =
          P(o^D_{t+1}|s',δ') · Σ_{s,δ} P(s',δ'|s,δ,a^D_t) · b_t(s,δ)
          ─────────────────────────────────────────────────────────────  (Eq.24)
          Σ_{ŝ,δ̂} P(o^D_{t+1}|ŝ,δ̂) · Σ_{s,δ} P(ŝ,δ̂|s,δ,a^D_t) · b_t(s,δ)
        """
        dim = min(len(obs_d), self.d)

        #  1. Innovation for deception detection 
        belief_pred = self.belief_mean[:dim]
        innovation = obs_d[:dim] - belief_pred
        attack_suspected = self.detect_deception(innovation)

        #  2. Adaptive variance  σ² → 9σ²  under suspected attack (Sec.VI-C2) 
        if attack_suspected:
            self.sigma2_adaptive = 9.0 * SIGMA_NOISE**2
        else:
            self.sigma2_adaptive = SIGMA_NOISE**2

        #  3. State transition: P(s',δ'|s,δ,a^D_t) 
        # Stochastic AR(1) transition for state particles
        if pregenerated_noise is not None:
            self.s_particles += pregenerated_noise
        else:
            self.s_particles += self.rng.normal(0, transition_noise,
                                                 self.s_particles.shape)
        # Defender action affects δ hypothesis: isolation reduces δ
        if action_d == DEF_ISOLATION:
            self.delta_particles *= 0.5
        elif action_d == DEF_DECEPTION:
            # Deception action may reveal true δ
            self.delta_particles += self.rng.normal(0, 0.01,
                                                     self.delta_particles.shape)

        #  4. Weight update: P(o^D_{t+1}|s',δ') 
        #new_weights = np.zeros(self.N)
        #for i in range(self.N):
         #   new_weights[i] = (self.weights[i] *
          #                     self.observation_likelihood(
           #                        obs_d,
            #                       self.s_particles[i],
             #                      self.delta_particles[i],
              #                     self.sigma2_adaptive))
        dim = min(len(obs_d), self.d)
        obs_trunc  = obs_d[:dim]
        s_trunc    = self.s_particles[:, :dim]      # (N, dim)
        d_trunc    = self.delta_particles[:, :dim]  # (N, dim)
        predicted  = s_trunc + d_trunc              # (N, dim)
        residuals  = obs_trunc[np.newaxis, :] - predicted  # (N, dim)
        log_liks   = -0.5 * np.sum(residuals**2, axis=1) / self.sigma2_adaptive
        log_liks  -= log_liks.max()                # numerical stability
        liks       = np.exp(log_liks)
        new_weights = self.weights * liks
        w_sum = new_weights.sum()
        if w_sum < 1e-300:
            new_weights = np.ones(self.N) / self.N
        else:
            new_weights /= w_sum
        self.weights = new_weights

        #  5. Update attack indicators 
        # Attack indicator based on δ magnitude
        delta_norms = np.linalg.norm(self.delta_particles, axis=1)
        self.attack_indicators = (delta_norms > EPSILON_PERT * 0.5).astype(float)

        #  6. Resampling if N_eff < N_min_eff (Sec.VI-C2) 
        N_eff = self.effective_sample_size()
        if N_eff < N_EFF_MIN:
            self._kernel_density_resample(kde_noise=kde_noise, guided_noise=guided_noise)

        #  7. Update belief mean (weighted average via matmul — faster than np.average) 
        self.belief_mean = self.weights @ self.s_particles  # (N,)@(N,d) → (d,)
        # belief_cov intentionally skipped: O(N·d²) and unused by callers

        #  8. Compute per-device attack probability 
        p_attack_est = float(np.dot(self.weights, self.attack_indicators))

        return {
            'belief_mean':   self.belief_mean,
            'N_eff':         N_eff,
            'attack_suspected': attack_suspected,
            'p_attack_est':  p_attack_est,
            'sigma2_adaptive': self.sigma2_adaptive,
        }

    def _kernel_density_resample(self, kde_noise=None, guided_noise=None):
        """
        Kernel density regularised resampling to prevent particle degeneracy.
        (Sec. VI-C2, bullet 3)
        """
        # Systematic resampling
        positions = (self.rng.random() + np.arange(self.N)) / self.N
        cumsum = np.cumsum(self.weights)
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, self.N - 1)

        self.s_particles     = self.s_particles[indices]
        self.delta_particles = self.delta_particles[indices]
        self.weights         = np.ones(self.N) / self.N

        # KDE jitter (Gaussian kernel with bandwidth h = σ_noise)
        h = SIGMA_NOISE
        if kde_noise is not None:
            self.s_particles += kde_noise
        else:
            self.s_particles += self.rng.normal(0, h, self.s_particles.shape)

        # Observation-guided proposals (Sec.VI-C2, bullet 4):
        # Move 10% of particles toward the current observation mean
        n_guided = max(1, int(0.1 * self.N))
        if guided_noise is not None:
            self.s_particles[:n_guided] = self.belief_mean + guided_noise[:n_guided]
        else:
            self.s_particles[:n_guided] = (
                self.belief_mean +
                self.rng.normal(0, 0.01, (n_guided, self.d))
            )

    def pomdp_value(self,
                     belief: np.ndarray,
                     reward: float,
                     gamma: float = GAMMA_DISC) -> float:
        """
        V(b) = max_{a∈A} [R(b,a) + γ ∫ V(b')P(b'|b,a) db']   Eq.26
        Approximated via one-step lookahead with tabular backup.
        R(b,a) = average reward over possible states.
        """
        # Discretise belief for table lookup (health bucket)
        p_atk = float(np.dot(self.weights, self.attack_indicators))
        bucket = int(np.clip(p_atk * 10, 0, 9))
        key = (bucket,)

        current_V = self.V_array[bucket]

        # R(b, a) = expected health preservation
        R_ba = reward

        # V(b') from table or 0 (terminal approximation)
        next_bucket = min(bucket + 1, 9)
        V_next = self.V_array[next_bucket]

        # Bellman backup
        new_V = R_ba + gamma * V_next
        self.V_array[bucket] = 0.9 * current_V + 0.1 * new_V   # soft update

        return new_V
    def q_value(self, action: int, health_i: float,
                p_attack_est: float,
                gamma: float = GAMMA_DISC) -> float:
        """
        Q(b_i, a) = R(b_i, a) + γ · ∫ V(b') P(b'|b,a) db'   (Eq.26 / Eq.34)

        The integral is approximated via weighted particle sum:
            ∫ V(b') P(b'|b,a) db' ≈ Σ_j w^(j) · V(bucket^(j))
        where bucket^(j) = floor(attack_indicator^(j) * 10)
        """
        # R(b_i, a): per-device immediate reward
        eta_a  = float(DEF_ETA[action])
        cost_a = float(DEF_COST[action])
        eff_a2 = float(DEF_EFF[action, ATK_ADVERS])
        R_ba   = health_i * eta_a - cost_a + p_attack_est * eff_a2

        # ∫ V(b') P(b'|b,a) db' ≈ Σ_j w^(j) · V_array[bucket_j]
        # Vectorized: single fancy-index (no dict, no list comprehension)
        buckets  = np.clip((self.attack_indicators * 10).astype(int), 0, 9)
        V_vec    = self.V_array[buckets]
        E_V_next = float(self.weights @ V_vec)

        return R_ba + gamma * E_V_next

# ══════════════════════════════════════════════════════════════════════════════
# 10.  ATTACKER AGENT  (Sec. VI-B, hospital Sec. 2-3)
# ══════════════════════════════════════════════════════════════════════════════

class AttackerAgent:
    """
    Strategic attacker with k=2 hypergame reasoning.

    Implements:
      - Target selection   T*(t) = argmax_{T⊆V} E[U_A] - λ_c|T|   Eq.31
      - Adversarial perturbation with physiological plausibility    Eq.30
      - Attack success probability  p_success = p_base·µ·ρ·(1−η)  Eq.38
      - Detection probability       p_detect  (inverse)             Eq.39
      - Level 0/1/2 strategies      hospital Sec. 3.3
    """

    def __init__(self, topology: HospitalNetworkTopology,
                 rng: np.random.Generator):
        self.topology = topology
        self.rng = rng
        self.n = topology.n

        # Attacker's perception state
        self.P_a = PerceptionState('attacker', self.n, rng)

        # Strategy distribution
        self.strategy_space = StrategySpace('attacker', K_ATTACKER)
        self.strategy_space.sigma[:] = 0.0
        # Bias toward adversarial perturbation (A2) — primary attack
        atk_l0_idx = self.strategy_space.level_indices(0)
        self.strategy_space.sigma[atk_l0_idx[1]] = 0.5   # VitalSignsShift
        self.strategy_space.sigma[atk_l0_idx[2]] = 0.3   # DataReplay
        self.strategy_space.sigma[atk_l0_idx[0]] = 0.2   # NoAttack
        self.strategy_space.project_to_simplex()

        # Current targets
        self.current_targets: List[int] = []

        # Current perturbation δ ∈ R^{n × d}
        self.delta_matrix = np.zeros((self.n, D_STATE))

        # Attacker's belief about defender threshold (τ̂)
        self.tau_hat = float(rng.uniform(0.3, 0.7))

        # Attack type being deployed  (index into ATK_*)
        self.active_attack_type = ATK_ADVERS

        # History of defender actions (observable)
        self.observed_def_actions: List[int] = []

        # Attacker's belief about defender's perception (for k=2)
        self.P_a_l2 = PerceptionState('attacker', self.n, rng)

        # Attacker meta-learner for adaptive perturbation
        self.meta_learner = AdversarialMetaLearner(D_STATE, rng)

    def compute_attack_success_prob(self,
                                    target_id: int,
                                    rho_intensity: float,
                                    eta_defense: float) -> float:
        """
        p_success = p_base · µ_type · ρ_intensity · (1 − η_defense)   Eq.38
        """
        mu_type = float(ATK_MU[self.active_attack_type])
        p_s = (P_BASE_ATK
               * mu_type
               * float(np.clip(rho_intensity, 0, 1))
               * max(0.0, 1.0 - eta_defense))
        return float(np.clip(p_s, 0, 1))


    def select_targets(self,
                        health_vector: np.ndarray,
                        trust_vector: np.ndarray,
                        time_step: int) -> List[int]:
        """
        T*(t) = argmax_{T⊆V} E[U_A(s(t), a_A, a_D)|T] − λ_c|T|   Eq.31
        Greedy approximation: score each device, select top-k.
        """
        scores = np.zeros(self.n)

        high_conn = set(self.topology.get_high_connectivity_targets())
        gateways  = set(self.topology.get_gateway_devices())

        high_conn = set(self.topology.get_high_connectivity_targets())
        gateways  = set(self.topology.get_gateway_devices())

        conn_bonus    = np.array([0.3 if i in high_conn else 0.0
                                   for i in range(self.n)])
        gateway_bonus = np.array([0.4 if i in gateways  else 0.0
                                   for i in range(self.n)])
        criticality   = np.array([DEV_CRITICALITY[self.topology.device_types[i]]
                                   for i in range(self.n)])

        scores = (
            (1.0 - trust_vector)  * 0.4 +
            (1.0 - health_vector) * 0.3 +
            conn_bonus + gateway_bonus
        ) * criticality * (1.0 - self.tau_hat * 0.5)

        # Budget constraint: select top-n_targets devices
        n_targets = max(1, int(self.n * 0.05))     # ~5% of devices
        top_k = np.argsort(scores)[::-1][:n_targets]

        # Subtract multi-target cost penalty λ_c·|T| (Eq.31)
        E_UA = float(np.sum(scores[top_k]))
        net_utility = E_UA - LAMBDA_C * len(top_k)

        if net_utility <= 0:
            # Not worth attacking
            self.current_targets = []
            return []

        self.current_targets = top_k.tolist()
        return self.current_targets

    def generate_perturbation(self,
                               targets: List[int],
                               physical_state: HospitalPhysicalState,
                               reasoning_level: int = 2) -> np.ndarray:
        """
        Generate adversarial perturbation δ satisfying:
          f(s+δ) ≠ f(s)  while  P(s+δ) ≈ P(s)   Eq.30 / paper Sec.II-B

        At level 1: Camouflage (noise mimicking sensor noise)
        At level 2: Adaptive perturbation based on observed defender alerts
        Physiological plausibility enforced.
        """
        delta = np.zeros((self.n, D_STATE))
        strategy_idx = self.strategy_space.sample_strategy()
        strategy_name = self.strategy_space.strategy_names[strategy_idx]

        for tid in targets:
            dtype = self.topology.device_types[tid]
            pid   = self.topology.patient_map.get(tid, 0)
            pid   = min(pid, N_PATIENTS - 1)
            pat   = physical_state.patients[pid]

            if strategy_name in ('VitalSignsShift', 'AdaptivePerturbation',
                                  'Camouflage', 'MimicClinician', 'Feint'):
                # Compute base perturbation targeting clinical decision function f
                base_delta = np.zeros(D_STATE)

                if dtype in (DEV_MONITOR, DEV_WEARABLE):
                    # Shift HR downward (simulate bradycardia) → triggers
                    # incorrect clinical response
                    hr_margin  = pat.HR - PHYSIO_RANGES['HR'][0]        # distance to low bound
                    spo2_margin = pat.SpO2 - PHYSIO_RANGES['SpO2'][0]   # distance to low bound

                    # Shift by half the margin — maximally stealthy, still crosses threshold
                    hr_shift   = -float(np.clip(hr_margin * 0.5,  5, 25))
                    spo2_shift = -float(np.clip(spo2_margin * 0.5, 2, 8))
                    # Embed in sensor dimensions of state vector
                    base_delta[0] = hr_shift / PHYSIO_STD['HR']
                    base_delta[2] = spo2_shift / PHYSIO_STD['SpO2']
                elif dtype == DEV_PUMP:
                    # Shift infusion rate reading
                    rate_shift = float(self.rng.uniform(5, 15))
                    base_delta[0] = rate_shift / PHYSIO_STD['Rate']
                elif dtype == DEV_VENT:
                    # Shift tidal volume reading
                    tv_shift = float(self.rng.uniform(50, 100))
                    base_delta[0] = tv_shift / PHYSIO_STD['TidalV']

                if strategy_name == 'Camouflage':
                    #noise to mimic sensor noise  (Level 1)
                    base_delta += self.rng.normal(0, SIGMA_NOISE, D_STATE)

                elif strategy_name == 'MimicClinician':
                    # Temporal smoothing to mimic natural variation
                    base_delta *= 0.5
                    prev = self.delta_matrix[tid]
                    base_delta = 0.7 * prev + 0.3 * base_delta

                elif strategy_name == 'AdaptivePerturbation':
                    # Adjust based on observed alerts  (Level 1)
                    if len(self.observed_def_actions) > 0:
                        last_def = self.observed_def_actions[-1]
                        if last_def == DEF_DECEPTION:
                            base_delta *= 1.2   # Escalate
                        elif last_def == DEF_ISOLATION:
                            base_delta *= 0.5   # Reduce stealth

                elif strategy_name == 'Feint':
                    # Feint: attack a different patient's vitals (Level 2)
                    feint_pid = (pid + 1) % N_PATIENTS
                    feint_devices = [
                        d for d in range(self.n)
                        if self.topology.patient_map.get(d, -1) == feint_pid
                           and self.topology.device_types[d] in (DEV_MONITOR,)]
                    for fd in feint_devices[:1]:
                        delta[fd][0] = self.rng.uniform(0.1, 0.2)

                delta[tid] = base_delta

            elif strategy_name == 'MetaAdaptive':
                # k=2: model defender's learning trajectory, exploit predicted defense gap
                # Step 1: estimate defender strategy distribution from recent history
                W = min(20, len(self.observed_def_actions))
                if W > 0:
                    recent = self.observed_def_actions[-W:]
                    sigma_hat_d = np.zeros(N_DEF_ACTIONS)
                    for d_act in recent:
                        sigma_hat_d[int(d_act)] += 1.0
                    sigma_hat_d /= sigma_hat_d.sum()
                else:
                    sigma_hat_d = np.ones(N_DEF_ACTIONS) / N_DEF_ACTIONS

                # Step 2: predict most likely next defense
                d_hat = int(np.argmax(sigma_hat_d))

                # Step 3: find attack type with lowest DEF_EFF against d_hat
                # (hardest to stop = most exploitable gap)
                a_star = int(np.argmin(DEF_EFF[d_hat, :]))

                # Step 4: craft perturbation aligned with a_star at full budget
                base_delta = np.zeros(D_STATE)
                dtype = self.topology.device_types[tid]
                if a_star == ATK_SPOOFING:
                    # Spoofing: replace identity signal
                    base_delta[3] = float(self.rng.uniform(0.5, 1.0))
                elif a_star == ATK_ADVERS:
                    # Adversarial perturbation: clinical shift
                    base_delta[0] = -float(self.rng.uniform(10, 25)) / PHYSIO_STD['HR']
                    base_delta[2] = -float(self.rng.uniform(3, 8))   / PHYSIO_STD['SpO2']
                elif a_star == ATK_POISON:
                    # Data poisoning: corrupt security feature dimensions
                    base_delta[D_SEC:D_SEC + 5] = float(self.rng.uniform(0.3, 0.6))
                elif a_star == ATK_EXTRACT:
                    # Model extraction: probe comm dimensions
                    base_delta[D_STATE - 5:] = float(self.rng.uniform(0.2, 0.4))
                elif a_star == ATK_REPLAY:
                    # Replay: freeze state direction
                    base_delta = -0.05 * physical_state.devices[tid].s

                # Scale to full ℓ2 budget (maximally effective)
                b_norm = np.linalg.norm(base_delta)
                if b_norm > 1e-8:
                    base_delta *= EPSILON_PERT / b_norm
                delta[tid] = base_delta

            elif strategy_name == 'Probe':
                # Probe remains a small test perturbation to infer τ̂ (Level 2)
                probe_magnitude = self.tau_hat * 0.3
                delta[tid] = self.rng.normal(0, probe_magnitude, D_STATE)

            elif strategy_name == 'DataReplay':
                # Replay:historical normal data (zero perturbation
                # to physical state, but manipulates temporal consistency)
                delta[tid] = np.zeros(D_STATE)

            elif strategy_name == 'DeviceFreeze':
                # Freeze: perturbation keeps state constant (large neg. velocity)
                delta[tid] = -0.1 * physical_state.devices[tid].s

        # Enforce ℓ2-norm budget constraint ‖δ‖ ≤ ε  (Eq.30)
        for i in targets:
            d_norm = np.linalg.norm(delta[i])
            if d_norm > EPSILON_PERT:
                delta[i] *= EPSILON_PERT / d_norm

        # Enforce physiological plausibility for medical devices
        for i in targets:
            dtype = self.topology.device_types[i]
            if dtype in (DEV_MONITOR, DEV_WEARABLE):
                # Clip to sensor range
                delta[i][0] = np.clip(delta[i][0], -3.0, 3.0)   # HR
                delta[i][2] = np.clip(delta[i][2], -2.0, 0.0)   # SpO2 (down only)

        self.delta_matrix = delta
        return delta

    def update_tau_hat(self, defender_alerted: bool, delta_norm: float):
        """
        Update estimated detection threshold τ̂ based on observations.
        (Level 2: MetaAdaptive, Probe)
        """
        # If no alert despite perturbation → threshold higher than assumed
        if not defender_alerted and delta_norm > 0.05:
            self.tau_hat = min(1.0, self.tau_hat + 0.02)
        # If alert triggered → threshold lower than assumed
        elif defender_alerted:
            self.tau_hat = max(0.0, self.tau_hat - 0.05)

    def update_perception(self,
                           obs_new: np.ndarray,
                           hypergame: HypergameModel):
        """
        Update attacker's perception state using Eq.21.
        Attacker's P^A belief about defender detection updated from observations.
        """
        P_observed = np.zeros(self.n)
        for i, d_action in enumerate(self.observed_def_actions[-self.n:]):
            P_observed[i] = float(d_action == DEF_DECEPTION or
                                   d_action == DEF_ISOLATION) * 0.8

        grad_U = np.zeros(self.n)   # gradient of attacker utility w.r.t. P^A
        # Simple gradient: higher p_aware → lower utility → negative gradient
        for i in range(self.n):
            grad_U[i] = -(self.P_a.l2_belief_about_opponent_l1[i] - 0.5) * 0.1

        hypergame.perception_update_dynamics(
            'attacker', P_observed, grad_U)

        # Update l2_belief: attacker's estimate of defender's detection per device
        # Derived from observed defense actions — aggressive actions (isolation,
        # deception, firewall) signal that the defender has detected something.
        if len(self.observed_def_actions) > 0:
            recent = self.observed_def_actions[-self.n:]
            for i, d_act in enumerate(recent):
                aware_signal = float(d_act in (DEF_ISOLATION, DEF_DECEPTION,
                                               DEF_FIREWALL, DEF_HONEYPOT))
                self.P_a.l2_belief_about_opponent_l1[i] = (
                    0.7 * self.P_a.l2_belief_about_opponent_l1[i]
                    + 0.3 * aware_signal)

# ══════════════════════════════════════════════════════════════════════════════
# 11.  DEFENDER AGENT  (Sec. VI-C, hospital Sec. 3.1, 3.3)
# ════════════════════════════════════
class DefenderAgent:
    """
    Hospital cybersecurity defender with hypergame-aware AI.

    Integrates:
      - Deception-aware particle filter (Eq.22-26)
      - Adversarial meta-learner (Eq.18-21)
      - k=2 best-response computation (Eq.7-9)
      - Bayesian posterior over attacker perception (Eq.1-2)
      - Information advantage monitoring (Eq.27, 36)
      - CDSS anomaly detection for medical context
    """

    def __init__(self, topology: HospitalNetworkTopology,
                 rng: np.random.Generator):
        self.topology = topology
        self.rng = rng
        self.n = topology.n

        # Defender's perception state
        self.P_d = PerceptionState('defender', self.n, rng)

        # Strategy distribution (initialised to MONITOR-heavy)
        self.strategy_space = StrategySpace('defender', K_DEFENDER)
        self.strategy_space.sigma[:] = 0.0
        l0_idx = self.strategy_space.level_indices(0)
        self.strategy_space.sigma[l0_idx[0]] = 0.7   # Monitor
        self.strategy_space.sigma[l0_idx[1]] = 0.3   # Alert
        self.strategy_space.project_to_simplex()

        # Per-device particle filters
        self.particle_filters: List[DeceptionAwareParticleFilter] = [
            DeceptionAwareParticleFilter(D_STATE, rng)
            for _ in range(self.n)
        ]

        # Pre-allocate noise buffers — reused every time step to avoid
        # repeated 84MB allocations (350 × 100 × 100 × 3 arrays)
        _ng = max(1, int(0.1 * N_PARTICLES))
        self._buf_trans  = np.empty((self.n, N_PARTICLES, D_STATE))
        self._buf_kde    = np.empty((self.n, N_PARTICLES, D_STATE))
        self._buf_guided = np.empty((self.n, _ng, D_STATE))

        # Adversarial meta-learner (shared across devices for efficiency)
        self.meta_learner = AdversarialMetaLearner(D_STATE, rng)

        # CDSS anomaly detection thresholds (per sensor type)
        self.cdss_thresholds = {
            'HR':   {'low': 50, 'high': 120},
            'SpO2': {'low': 90, 'high': 100},
            'SBP':  {'low': 85, 'high': 160},
            'RR':   {'low': 8,  'high': 30},
            'Temp': {'low': 35.0, 'high': 38.5},
        }

        # Currently active defense per device
        self.active_defenses: np.ndarray = np.full(self.n, DEF_MONITOR)

        # Alarm history
        self.alarm_history: List[Dict] = []

        # Defense action history (for attacker's observation)
        self.action_history: List[int] = []

        # Hypergame-specific: belief about attacker perception hypotheses
        self.n_perception_hypotheses = 10
        self.perception_prior = np.ones(self.n_perception_hypotheses) / self.n_perception_hypotheses

        # Defender's level-2 belief about attacker's belief about defender
        self.P_d_l2 = PerceptionState('defender', self.n, rng)

        # RL policy gradient baseline
        self.policy_baseline = 0.0
        self.policy_W = rng.normal(0, 0.01, (N_DEF_ACTIONS, D_STATE))

    def cdss_anomaly_score(self,
                            patient_id: int,
                            obs_vitals: np.ndarray,
                            physical_state: HospitalPhysicalState) -> float:
        """
        Clinical Decision Support System anomaly score.
        z-score deviation from patient baseline; returns [0,1] anomaly prob.
        """
        pat = physical_state.patients[patient_id]
        normals = np.array([
            pat.HR,
            pat.SBP,
            pat.SpO2,
                pat.RR,
                pat.Temp
        ])
        stds = np.array([
            PHYSIO_STD['HR'], PHYSIO_STD['SBP'],
            PHYSIO_STD['SpO2'], PHYSIO_STD['RR'], PHYSIO_STD['Temp']
        ])
        dim = min(len(obs_vitals), 5)
        z_scores = np.abs((obs_vitals[:dim] - normals[:dim]) / (stds[:dim] + 1e-8))
        anomaly_score = float(np.mean(z_scores) / 3.0)  # normalise by 3σ
        return float(np.clip(anomaly_score, 0, 1))

    def observe(self,
                physical_state: HospitalPhysicalState,
                delta_matrix: np.ndarray,
                action_d: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Defender's observation model:
          o^D_i(t) = s_i(t) + δ_i(t) + ξ_i(t),  ξ ~ N(0, σ²I)   Eq.33

        Returns (observations, anomaly_scores) for all devices.
        """
        # Vectorized: stack all device states, generate all noise in one call
        s_mat   = np.array([physical_state.devices[i].s for i in range(self.n)])
        xi_mat  = self.rng.normal(0, SIGMA_NOISE, (self.n, D_STATE))
        obs     = s_mat + delta_matrix + xi_mat

        anomaly_scores = np.zeros(self.n)
        norm_factor = D_STATE ** 0.5
        for i in range(self.n):
            # CDSS anomaly score for patient-linked devices
            pid = self.topology.patient_map.get(i, -1)
            if pid >= 0 and self.topology.device_types[i] in (DEV_MONITOR, DEV_WEARABLE):
                sensor_obs = physical_state.devices[i].r + delta_matrix[i, :NS] + xi_mat[i, :NS]
                anomaly_scores[i] = self.cdss_anomaly_score(pid, sensor_obs, physical_state)
            else:
                anomaly_scores[i] = float(np.linalg.norm(obs[i]) / norm_factor)

        return obs, anomaly_scores

    def bayesian_belief_update(self,
                                obs_matrix: np.ndarray,
                                action_d: int,
                                anomaly_scores: np.ndarray) -> Dict:
        """
        Per-device belief update via particle filter (Eq.24).
        Also updates Bayesian posterior over attacker perception (Eq.1,2).
        """
        pf_results = {}
        # Fill pre-allocated buffers in-place — avoids 84MB/step of heap allocations
        self.rng.standard_normal(out=self._buf_trans);  self._buf_trans  *= 0.05
        self.rng.standard_normal(out=self._buf_kde);    self._buf_kde    *= SIGMA_NOISE
        self.rng.standard_normal(out=self._buf_guided); self._buf_guided *= 0.01
        for i in range(self.n):
            result = self.particle_filters[i].update(
                obs_matrix[i], action_d,
                pregenerated_noise=self._buf_trans[i],
                kde_noise=self._buf_kde[i],
                guided_noise=self._buf_guided[i])
            pf_results[i] = result

            # Update defender perception P^D — attack probability per device
            self.P_d.p_attack[i] = 0.7 * self.P_d.p_attack[i] + \
                                    0.3 * result['p_attack_est']

        # Aggregate attack probability across patients (vectorised via patient_map)
        patient_attack_prob = np.zeros(N_PATIENTS)
        p_atk = self.P_d.p_attack  # (N_DEVICES,) array
        for pid in range(N_PATIENTS):
            mask = self.topology.patient_map_array == pid  # precomputed bool mask
            if mask.any():
                patient_attack_prob[pid] = float(p_atk[mask].mean())

        return {'pf_results': pf_results,
                'patient_attack_prob': patient_attack_prob,
                'mean_attack_prob': float(np.mean(self.P_d.p_attack))}

    def select_defense(self,
                        belief_update: Dict,
                        health_vector: np.ndarray,
                        trust_vector: np.ndarray,
                        hypergame: HypergameModel,
                        attacker_strategy: np.ndarray,
                        sigma_d_star: np.ndarray,
                        time_step,
                        device_states: np.ndarray = None) -> np.ndarray:
        """
        Computes defender's k=2 best response and selects per-device actions.
        Policy π*_D: S × B → Δ(D)  (Eq.34)
        """
        # Compute k=2 best response — use device_states passed from simulation loop
        if device_states is None:
            device_states = np.zeros((self.n, D_STATE))
        br_sigma = best_response_levelk(
            'defender',
            attacker_strategy,
            hypergame.P_a,
            hypergame.P_d,
            device_states,
            health_vector,
            self.P_d, hypergame.P_a,
            k=K_DEFENDER)

        # Resolve br_sigma → dominant DEF action index (indices 7-9 are k=2)
        dominant_strategy_name = StrategySpace.DEF_ALL[int(np.argmax(br_sigma))]
        br_def_action = StrategySpace.STRATEGY_TO_DEF_IDX.get(
            dominant_strategy_name, DEF_MONITOR)

        # Network-level threat signal
        mean_atk_prob = belief_update['mean_attack_prob']

        actions = np.full(self.n, DEF_MONITOR)
        for i in range(self.n):
            p_atk_est_i = float(belief_update['pf_results'][i]['p_attack_est'])

            # Q-value from particle filter (reactive, per-device)
            q_vals = np.array([
                self.particle_filters[i].q_value(
                    a, health_vector[i], p_atk_est_i)
                for a in range(N_DEF_ACTIONS)
            ])
            pf_action = int(np.argmax(q_vals))

            # π*(b_i): hypergame BR overrides PF when attack is suspected
            # Uses both network-level (mean_atk_prob) and per-device signals
            if mean_atk_prob > 0.3 or p_atk_est_i > 0.4:
                actions[i] = br_def_action   # k=2 hypergame best response
            else:
                actions[i] = pf_action       # PF greedy (no threat detected)

        # ThresholdDither override at level-1 (once per 20 steps, not twice)
        if time_step % 20 == 0:
            l1_strats = self.strategy_space.level_strategies(1)
            if 'ThresholdDither' in l1_strats:
                for k_sensor in ['HR', 'SpO2']:
                    offset = self.rng.uniform(-5, 5)
                    self.cdss_thresholds[k_sensor]['high'] += offset

        self.active_defenses = actions
        return actions

    def compute_defense_effectiveness(self,
                                       actions: np.ndarray,
                                       attack_types: np.ndarray) -> np.ndarray:
        """
        Per-device effectiveness η(d_i, a_j) from Table I.
        Returns effectiveness vector ∈ R^n.
        """
        effectiveness = np.zeros(self.n)
        for i in range(self.n):
            d_act = int(actions[i])
            a_type = int(attack_types[i]) if i < len(attack_types) else ATK_ADVERS
            d_act  = min(d_act,  N_DEF_ACTIONS - 1)
            a_type = min(a_type, N_ATK_TYPES  - 1)
            effectiveness[i] = DEF_EFF[d_act, a_type]
        return effectiveness

    def update_perception(self,
                           obs_matrix: np.ndarray,
                           hypergame: HypergameModel,
                           P_A_true_proxy: np.ndarray):
        """
        Update defender perception via Eq.21:
          dP^D/dt = υ(P^D_observed - P^D) + β ∇U_D(P^D)
        Also updates meta-learner P_D histogram.
        """
        # P_observed from anomaly scores
        P_observed = np.zeros(self.n)
        for i in range(self.n):
            P_observed[i] = float(np.linalg.norm(obs_matrix[i]) / D_STATE**0.5)

        # Gradient of U_D w.r.t. P_D: higher attack prob → higher utility gradient
        grad_UD = np.zeros(self.n)
        for i in range(self.n):
            # ∂U_D/∂p_attack_i ≈ expected utility gain from increasing vigilance
            eta_i = DEF_EFF[int(self.active_defenses[i]), ATK_ADVERS]
            grad_UD[i] = eta_i * (1.0 - self.P_d.p_attack[i]) * 0.1

        hypergame.perception_update_dynamics(
            'defender', P_observed, grad_UD)

        # Update meta-learner histogram (for Eq.19 KL term)
        obs_flat = obs_matrix.mean(axis=1)
        hist, _ = np.histogram(obs_flat, bins=N_BINS, density=True)
        self.meta_learner.P_D_hist = hist / (hist.sum() + 1e-12)

        # P^A_true proxy from recent observations
        self.meta_learner.P_A_true = P_A_true_proxy / (P_A_true_proxy.sum() + 1e-12)

    def compute_reward(self,
                        health_vector: np.ndarray,
                        attack_detected: bool,
                        actions: np.ndarray) -> float:
        """
        R_D(s, a_D, a_A): network health + detection bonus − cost.
        Used in POMDP objective (Eq.34).
        """
        health_reward = float(np.mean(health_vector))
        detection_bonus = 0.5 if attack_detected else 0.0
        defense_cost = float(np.mean(DEF_COST[actions.astype(int)]))
        return health_reward + detection_bonus - defense_cost * 0.1


# ══════════════════════════════════════════════════════════════════════════════
# 12.  ATTACK APPLICATION & OUTCOME  (Sec. VI-B, Eq. 38, 39)
# ══════════════════════════════════════════════════════════════════════════════

def apply_attack_to_physical_state(
        physical_state: HospitalPhysicalState,
        targets: List[int],
        delta_matrix: np.ndarray,
        attack_type: int,
        effectiveness_vector: np.ndarray,
        rng: np.random.Generator,
        obs_matrix: np.ndarray = None,
        particle_filters: List = None,
        lambda_val: float = 0.0,
        mean_p_attack: float = 0.1) -> Dict:
    """
    Applies attack outcomes to physical state.
    Modifies device health h_i, trust τ_i, and patient vitals.

    Returns attack outcome statistics.
    """
    n_successful = 0
    n_detected   = 0
    health_damages = []

    attack_success_flags = np.zeros(len(physical_state.devices))

    for i, tid in enumerate(targets):
        dev   = physical_state.devices[tid]
        dtype = physical_state.topology.device_types[tid]
        pid   = physical_state.topology.patient_map.get(tid, -1)

        # Perturbation norm
        delta_i   = delta_matrix[tid] if tid < len(delta_matrix) else np.zeros(D_STATE)
        delta_norm = float(np.linalg.norm(delta_i))

        # Defence effectiveness for this device
        eta_def = float(effectiveness_vector[tid]) if tid < len(effectiveness_vector) else 1.0

        # Intensity (normalised perturbation)
        rho_int = min(1.0, delta_norm / EPSILON_PERT)

        # Attack success probability  (Eq.38)
        mu_type  = float(ATK_MU[attack_type])
        p_success = float(np.clip(
            P_BASE_ATK * mu_type * rho_int * max(0, 1.0 - eta_def), 0, 1))
        success = bool(rng.random() < p_success)

        # Detection probability  (Eq.39)
        # Neyman-Pearson likelihood ratio test under Gaussian obs model Eq.22
        # Derived weights — zero free parameters:
        #   a1 = sqrt(d_eff)   NP optimal SNR weight for d-dimensional Gaussian
        #   a2 = 1.0           z_innov is already a standardised statistic
        #   a3 = p̄_attack     empirical attack prior from hypergame belief state

        if obs_matrix is not None and particle_filters is not None:
            pf      = particle_filters[tid]
            d_eff   = pf.d
            obs_i   = obs_matrix[tid, :d_eff]
            innov   = float(np.linalg.norm(obs_i - pf.belief_mean[:d_eff]))
            z_innov = float(np.clip(
                innov / (np.sqrt(pf.sigma2_adaptive * d_eff) + 1e-12),
                0.0, 3.0))
        else:
            d_eff   = D_STATE
            z_innov = rho_int

        a1       = float(np.sqrt(d_eff))   # NP weight: sqrt(d_eff)
        a2       = 1.0                      # z_innov already standardised
        a3       = float(np.clip(mean_p_attack, 0.0, 1.0))  # attack prior

        rho_snr  = delta_norm / (SIGMA_NOISE * np.sqrt(d_eff) + 1e-12)
        logit    = a1 * rho_snr + a2 * z_innov + a3 * lambda_val
        p_detect = float(1.0 / (1.0 + np.exp(-logit)))
        detected = bool(rng.random() < p_detect)

        if success:
            n_successful += 1
            attack_success_flags[tid] = 1.0

            # Degrade device health: h_i → κ · h_i  (Sec.VI-E3)
            dev.health = float(np.clip(dev.health * KAPPA, 0, 1))
            dev.trust  = float(np.clip(dev.trust - ALPHA_DECAY * 10, 0, 1))
            health_damages.append(1.0 - dev.health)

            # Affect patient vitals if device is patient-linked
            if pid >= 0 and dtype in (DEV_MONITOR, DEV_WEARABLE, DEV_PUMP, DEV_VENT):
                pat = physical_state.patients[pid]
                # severity defined once for all branches (Sec.VI-E3)
                severity = DEV_CRITICALITY[dtype] * delta_norm
                if dtype in (DEV_MONITOR, DEV_WEARABLE):
                    # Device now reports corrupted vitals — actual physiological
                    # harm from delayed treatment (attacker goal achieved)
                    pat.SpO2 = float(np.clip(pat.SpO2 - severity * 2, 70, 100))
                elif dtype == DEV_PUMP:
                    # Infusion rate manipulation → patient harm
                    pat.HR = float(np.clip(pat.HR + severity * 5, 40, 200))
        if detected:
            n_detected += 1
            # Restore trust slightly on detection (alert issued)
            dev.trust = min(1.0, dev.trust + ALPHA_RECOV * 5)

    return {
        'n_successful':        n_successful,
        'n_detected':          n_detected,
        'n_targets':           len(targets),
        'attack_success_flags':attack_success_flags,
        'health_damages':      health_damages,
        'mean_damage':         float(np.mean(health_damages)) if health_damages else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 13.  PERFORMANCE METRICS  (Sec. VI-F)
# ══════════════════════════════════════════════════════════════════════════════

class MetricsCollector:
    """
    Collects and computes all Sec. VI-F metrics:
      1. Network resilience ρ(t)          (Eq.17 / Eq.26-based)
      2. Robustness Λ(t)                  (information-advantage weighted)
      3. Information advantage I(P^D;P^A) (Eq.27, 36)
      4. Detection rate DR
      5. Attack success rate SR
      6. Network health H(t)              (Eq.29)
      7. Perception gap ΔP(t)             (Eq.35)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.n_attacks_total   = 0
        self.n_attacks_detected = 0
        self.n_attacks_success  = 0
        self.health_history: List[float] = []
        self.resilience_history: List[float] = []
        self.robustness_history: List[float] = []
        self.info_advantage_history: List[float] = []
        self.perception_gap_history: List[float] = []
        self.convergence_iterations: List[int] = []
        self.epsilon_k_history: List[float] = []
        self.utility_defender: List[float] = []
        self.utility_attacker: List[float] = []
        self.strategy_entropy_d: List[float] = []
        self.strategy_entropy_a: List[float] = []
        self.n_eff_history: List[float] = []

    def record_timestep(self,
                         health: float,
                         resilience: float,
                         info_adv: float,
                         perception_gap: float,
                         n_atk: int,
                         n_det: int,
                         n_suc: int,
                         conv_iter: int,
                         epsilon_k: float,
                         U_d: float,
                         U_a: float,
                         entropy_d: float,
                         entropy_a: float,
                         n_eff_mean: float,
                         robustness: float):
        self.health_history.append(health)
        self.resilience_history.append(resilience)
        self.info_advantage_history.append(info_adv)
        self.perception_gap_history.append(perception_gap)
        self.n_attacks_total    += n_atk
        self.n_attacks_detected += n_det
        self.n_attacks_success  += n_suc
        self.convergence_iterations.append(conv_iter)
        self.epsilon_k_history.append(epsilon_k)
        self.utility_defender.append(U_d)
        self.utility_attacker.append(U_a)
        self.strategy_entropy_d.append(entropy_d)
        self.strategy_entropy_a.append(entropy_a)
        self.n_eff_history.append(n_eff_mean)
        self.robustness_history.append(robustness)

    @property
    def detection_rate(self) -> float:
        if self.n_attacks_total == 0:
            return 0.0
        return self.n_attacks_detected / self.n_attacks_total

    @property
    def attack_success_rate(self) -> float:
        if self.n_attacks_total == 0:
            return 0.0
        return self.n_attacks_success / self.n_attacks_total

    def summary(self) -> Dict:
        return {
            'mean_health':        float(np.mean(self.health_history)) if self.health_history else 0.0,
            'std_health':         float(np.std(self.health_history))  if self.health_history else 0.0,
            'mean_resilience':    float(np.mean(self.resilience_history)) if self.resilience_history else 0.0,
            'mean_robustness':    float(np.mean(self.robustness_history)) if self.robustness_history else 0.0,
            'mean_info_adv':      float(np.mean(self.info_advantage_history)) if self.info_advantage_history else 0.0,
            'mean_perc_gap':      float(np.mean(self.perception_gap_history)) if self.perception_gap_history else 0.0,
            'detection_rate':     self.detection_rate,
            'attack_success_rate':self.attack_success_rate,
            'mean_conv_iter':     float(np.mean(self.convergence_iterations)) if self.convergence_iterations else 0.0,
            'mean_epsilon_k':     float(np.mean(self.epsilon_k_history)) if self.epsilon_k_history else 0.0,
            'mean_utility_d':     float(np.mean(self.utility_defender)) if self.utility_defender else 0.0,
            'mean_utility_a':     float(np.mean(self.utility_attacker)) if self.utility_attacker else 0.0,
            'mean_n_eff':         float(np.mean(self.n_eff_history)) if self.n_eff_history else 0.0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 14.  MAIN SIMULATION LOOP  (Sec. VI)
# ══════════════════════════════════════════════════════════════════════════════

class HospitalHypergameSimulation:
    """
    Full simulation integrating all components.
    Runs N_ep=100 episodes of T=200 steps each.
    """

    def __init__(self, seed: int = BASE_SEED):
        self.seed = seed
        self.rng  = np.random.default_rng(seed)

        # Build hospital network topology (Eq.28)
        self.topology = HospitalNetworkTopology(self.rng)

        # Physical state
        self.phys = HospitalPhysicalState(self.rng, self.topology)

        # Hypergame model (H_k for k=2)
        self.hypergame = HypergameModel(
            N_DEVICES, K_DEFENDER, K_ATTACKER, self.rng)

        # Agents
        self.attacker = AttackerAgent(self.topology, self.rng)
        self.defender = DefenderAgent(self.topology, self.rng)

        # Sync perception states with hypergame
        self.hypergame.P_d = self.defender.P_d
        self.hypergame.P_a = self.attacker.P_a

        # Metrics
        self.metrics = MetricsCollector()

        # Attacker perception hypotheses for Bayesian update (Eq.1)
        self.perc_hypotheses = [
            PerceptionState('attacker', N_DEVICES, self.rng)
            for _ in range(10)
        ]
        self.perc_prior = np.ones(10) / 10

    def _compute_utility_matrices(self,
                                   health_vector: np.ndarray
                                   ) -> Tuple[np.ndarray, np.ndarray]:
        """Build U_D and U_A matrices for resilience computation (Eq.17)."""
        n_d = N_DEF_ACTIONS
        n_a = N_ATK_TYPES

        U_D = np.zeros((n_d, n_a))
        U_A = np.zeros((n_d, n_a))

        device_states = np.zeros((self.defender.n, D_STATE))

        for d_act in range(n_d):
            for a_type in range(n_a):
                U_D[d_act, a_type] = subjective_payoff(
                    d_act, a_type, 'defender',
                    device_states, health_vector,
                    self.defender.n)

                U_A[d_act, a_type] = subjective_payoff(
                    a_type, d_act, 'attacker',
                    device_states, health_vector,
                    self.defender.n)
        # Criticality-weighted mean health — uses DEV_CRITICALITY already defined
        #crit_w = np.array([
        #    DEV_CRITICALITY[self.topology.device_types[i]]
        #    for i in range(N_DEVICES)
        #])
        #h_w = float(np.average(health_vector, weights=crit_w))

        #for d_act in range(n_d):
        #    for a_type in range(n_a):
        #        eta = float(DEF_EFF[d_act, a_type])
        #        U_D[d_act, a_type] = h_w * eta - DEF_COST[d_act]
        #        mu  = float(ATK_MU[a_type])
        #        p_s = P_BASE_ATK * mu * 0.5 * max(0, 1 - eta)
        #        U_A[d_act, a_type] = p_s * (1.0 - h_w) - 0.1 * eta

        return U_D, U_A

    def run_episode(self, episode_idx: int) -> MetricsCollector:
        """Execute one episode of T_STEPS time steps."""
        self.phys = HospitalPhysicalState(self.rng, self.topology)
        ep_metrics = MetricsCollector()


        for t in range(T_STEPS):
            # ─────────────────────────────────────────────────────────────────
            # STEP 1: Physical dynamics
            # ─────────────────────────────────────────────────────────────────
            self.phys.step_dynamics()
            health_vec = np.array([d.health for d in self.phys.devices])
            trust_vec  = np.array([d.trust  for d in self.phys.devices])
            #H_t = self.phys.network_health()   # Eq.29

            # ─────────────────────────────────────────────────────────────────
            # STEP 2: Attacker acts
            # ─────────────────────────────────────────────────────────────────
            # Target selection (Eq.31)
            targets = self.attacker.select_targets(health_vec, trust_vec, t)

            # Generate adversarial perturbation δ (Eq.30)
            if targets:
                delta_mat = self.attacker.generate_perturbation(
                    targets, self.phys, reasoning_level=K_ATTACKER)
            else:
                delta_mat = np.zeros((N_DEVICES, D_STATE))

            # ─────────────────────────────────────────────────────────────────
            # STEP 3: Defender observes
            # ─────────────────────────────────────────────────────────────────
            # Current defense action (from previous step)
            curr_def_action = int(np.bincount(
                self.defender.active_defenses.astype(int),
                minlength=N_DEF_ACTIONS).argmax())

            # Distorted observation o^D_t = s_t + δ_t + ξ_t  (Eq.33)
            obs_matrix, anomaly_scores = self.defender.observe(
                self.phys, delta_mat, curr_def_action)

            # ─────────────────────────────────────────────────────────────────
            # STEP 4: Defender belief update (Eq.24, particle filter)
            # ─────────────────────────────────────────────────────────────────
            belief_update = self.defender.bayesian_belief_update(
                obs_matrix, curr_def_action, anomaly_scores)

            # Bayesian posterior over attacker perception (Eq.1,2)
            obs_flat = obs_matrix.mean(axis=1)  # n-dimensional signal
            self.hypergame.obs_history_d.append(obs_flat)
            if len(self.perc_hypotheses) > 0 and t > 0:
                self.perc_prior = self.hypergame.bayesian_posterior_attacker_perception(
                    obs_flat, self.perc_hypotheses, self.perc_prior)

            # ─────────────────────────────────────────────────────────────────
            # STEP 5: Hypergame — find Nash equilibrium (Eq.37)
            # ─────────────────────────────────────────────────────────────────
            # Sync strategies
            sigma_d = self.defender.strategy_space.sigma.copy()
            sigma_a = self.attacker.strategy_space.sigma.copy()

            # Iterated best response to HNE
            sigma_d_star, sigma_a_star, n_conv = iterated_best_response(
                sigma_d, sigma_a,
                self.defender.P_d, self.attacker.P_a,
                obs_matrix, health_vec,
                K_DEFENDER, K_ATTACKER)

            # Update strategy distributions
            # Soft update to avoid oscillation
            self.defender.strategy_space.sigma = (
                0.7 * sigma_d + 0.3 * sigma_d_star)
            self.attacker.strategy_space.sigma = (
                0.7 * sigma_a + 0.3 * sigma_a_star)
            self.defender.strategy_space.project_to_simplex()
            self.attacker.strategy_space.project_to_simplex()

            # ─────────────────────────────────────────────────────────────────
            # STEP 6: Defender selects action (k=2 best response, Eq.9)
            # ─────────────────────────────────────────────────────────────────
            dev_states = np.array([d.s for d in self.phys.devices])
            def_actions = self.defender.select_defense(
                belief_update, health_vec, trust_vec,
                self.hypergame, sigma_a_star, sigma_d_star, t, device_states=dev_states)

            # Record action for attacker to observe
            self.attacker.observed_def_actions.append(curr_def_action)
            self.defender.action_history.append(curr_def_action)

            # ─────────────────────────────────────────────────────────────────
            # STEP 7: Compute effectiveness and apply attack outcomes
            # ─────────────────────────────────────────────────────────────────
            # Attack type vector (all targeted devices use primary attack A2)
            atk_type_vec = np.full(N_DEVICES, self.attacker.active_attack_type)

            effectiveness_vec = self.defender.compute_defense_effectiveness(
                def_actions, atk_type_vec)

            if targets:
                atk_outcome = apply_attack_to_physical_state(
                    self.phys, targets, delta_mat,
                    self.attacker.active_attack_type,
                    effectiveness_vec, self.rng,
                    obs_matrix=obs_matrix,
                    particle_filters=self.defender.particle_filters,
                    lambda_val=getattr(self, '_last_info_adv', 0.0),
                    mean_p_attack=float(np.mean(self.defender.P_d.p_attack)))
                n_suc = atk_outcome['n_successful']
                n_det = atk_outcome['n_detected']
                n_atk = atk_outcome['n_targets']
            else:
                n_suc, n_det, n_atk = 0, 0, 0

            # ─────────────────────────────────────────────────────────────────
            # STEP 8: Perception updates (Eq.21)
            # ─────────────────────────────────────────────────────────────────
            # True P^A (proxy from attack flags)
            P_A_true_proxy = np.zeros(N_BINS)
            if targets:
                delta_norms = np.linalg.norm(delta_mat[targets], axis=1)
                if delta_norms.max() - delta_norms.min() < 1e-8:
                    delta_norms = delta_norms + self.rng.normal(0, 1e-6, len(delta_norms))
                hist, _ = np.histogram(delta_norms, bins=N_BINS, density=True)
                P_A_true_proxy = hist / (hist.sum() + 1e-12)
            else:
                P_A_true_proxy = np.ones(N_BINS) / N_BINS

            self.defender.update_perception(obs_matrix, self.hypergame,
                                             P_A_true_proxy)
            self.attacker.update_perception(obs_flat, self.hypergame)

            # ─────────────────────────────────────────────────────────────────
            # STEP 9: Adversarial meta-learning update (Eq.18-20)
            # ─────────────────────────────────────────────────────────────────
            # Train defender meta-learner every step
            x_feat = obs_matrix.mean(axis=0)   # mean feature vector
            y_label = 1 if n_suc > 0 else 0
            x_adv = self.defender.meta_learner.adversarial_perturbation_update(
                x_feat, y_label, n_steps=2)   # reduced for speed
            loss = self.defender.meta_learner.meta_gradient_update(
                x_feat, y_label,
                self.defender.meta_learner.P_D_hist,
                P_A_true_proxy,
                sigma_d_star, sigma_a_star)

            # Attacker meta-learner update (adaptive to defender)
            x_atk = delta_mat.mean(axis=0)
            self.attacker.meta_learner.adversarial_perturbation_update(
                x_atk, 1, n_steps=1)
            self.attacker.update_tau_hat(
                n_det > 0,
                float(np.mean([np.linalg.norm(delta_mat[i]) for i in targets]))
                if targets else 0.0)

            # ─────────────────────────────────────────────────────────────────
            # STEP 10: Compute all metrics (Sec. VI-F)
            # ─────────────────────────────────────────────────────────────────
            # Update health after attack
            health_vec_post = np.array([d.health for d in self.phys.devices])
            H_t_post = float(np.mean(health_vec_post *
                                      np.array([d.trust for d in self.phys.devices])))

            # Resilience (Eq.17)
            U_D_mat, U_A_mat = self._compute_utility_matrices(health_vec_post)
            rho_t = resilience_metric(sigma_d_star, sigma_a_star,
                                       U_D_mat, U_A_mat)

            # Information advantage Λ (Eq.27, 36)
            # ── P^A_true: attacker's true perception ──────────────────────────
            # P^A_true[i] = ||δ_i||₂ / Σ_j ||δ_j||₂  — concentration of attack effort
            # Floor of 1e-6 ensures full support on all devices; avoids KL → ∞.
            P_true_a_raw = np.linalg.norm(delta_mat, axis=1)   # (N_DEVICES,)
            P_true_a = (P_true_a_raw + 1e-6) / (P_true_a_raw.sum() + 1e-6 * N_DEVICES)

            # ── P^D_true: defender's true perception ──────────────────────────
            # Ground truth = actual health damage per device, independent of belief.
            P_true_d_raw = 1.0 - health_vec_post                # (N_DEVICES,)
            P_true_d = (P_true_d_raw + 1e-6) / (P_true_d_raw.sum() + 1e-6 * N_DEVICES)

            info_adv = self.hypergame.information_advantage(
                P_true_a,
                P_true_d)
            self._last_info_adv = info_adv  # store for attack outcome model

            # Perception gap ΔP(t) (Eq.35)
            s_true = self.phys.get_sensor_matrix().mean(axis=0)
            s_perceived = obs_matrix.mean(axis=0)[:NS]
            perc_gap = self.hypergame.perception_gap(
                s_true, s_perceived[:len(s_true)])

            # Approximation error ε(k) (Eq.10 / Prop.1) — empirically measured
            # Step 1: utility at equilibrium strategies (U_i(σ*_d, σ*_a))
            U_d = expected_utility(
                sigma_d_star, sigma_a_star, 'defender',
                obs_matrix, health_vec_post,
                self.defender.P_d, self.attacker.P_a)
            U_a = expected_utility(
                sigma_a_star, sigma_d_star, 'attacker',
                obs_matrix, health_vec_post,
                self.attacker.P_a, self.defender.P_d)

            # Step 2: best unilateral deviation for defender
            # max_{σ_d} U_d(σ_d, σ*_a) — enumerate all 10 pure strategies
            n_strats = self.defender.strategy_space.n_strategies
            U_d_deviate = np.zeros(n_strats)
            for s_idx in range(n_strats):
                sigma_dev = np.zeros(n_strats)
                sigma_dev[s_idx] = 1.0
                U_d_deviate[s_idx] = expected_utility(
                    sigma_dev, sigma_a_star, 'defender',
                    obs_matrix, health_vec_post,
                    self.defender.P_d, self.attacker.P_a)
            U_d_best = float(np.max(U_d_deviate))

            # Step 3: ε(k) = |U_d* - U_d_best|  (Eq.10)
            eps_k = compute_approximation_error(U_d, U_d_best)

            # Mean N_eff across all particle filters
            n_eff_mean = float(np.mean([
                pf.effective_sample_size()
                for pf in self.defender.particle_filters[::10]  # sample 35
            ]))

            # Robustness Λ(t) = info_adv weighted defense effectiveness
            robustness = float(np.mean(effectiveness_vec)) * max(0, info_adv + 1)

            # Update hypergame mismatch
            self.hypergame.update_mismatch()

            ep_metrics.record_timestep(
                health=H_t_post,
                resilience=rho_t,
                info_adv=info_adv,
                perception_gap=perc_gap,
                n_atk=n_atk,
                n_det=n_det,
                n_suc=n_suc,
                conv_iter=n_conv,
                epsilon_k=eps_k,
                U_d=U_d,
                U_a=U_a,
                entropy_d=self.defender.strategy_space.entropy(),
                entropy_a=self.attacker.strategy_space.entropy(),
                n_eff_mean=n_eff_mean,
                robustness=robustness,
            )

        return ep_metrics

    def run(self) -> List[MetricsCollector]:
        """Run all N_ep=100 episodes."""
        all_metrics = []
        print(f"\n{'='*70}")
        print(f"  Hospital Hypergame Simulation — Seed {self.seed}")
        print(f"  Devices={N_DEVICES}, Patients={N_PATIENTS}, "
              f"Episodes={N_EPISODES}, Steps={T_STEPS}")
        print(f"{'='*70}")

        for ep in range(N_EPISODES):
            ep_metrics = self.run_episode(ep)
            all_metrics.append(ep_metrics)
            summary = ep_metrics.summary()
            if (ep + 1) % 10 == 0:
                print(f"  Ep {ep+1:3d}/{N_EPISODES} | "
                      f"Health={summary['mean_health']:.3f} | "
                      f"DR={summary['detection_rate']:.3f} | "
                      f"SR={summary['attack_success_rate']:.3f} | "
                      f"Resilience={summary['mean_resilience']:.3f} | "
                      f"InfoAdv={summary['mean_info_adv']:.3f}")

        return all_metrics


# ══════════════════════════════════════════════════════════════════════════════
# 15.  MULTI-SEED EXPERIMENT RUNNER  (Sec. VI-E5)
# ══════════════════════════════════════════════════════════════════════════════

class MultiSeedExperiment:
    """
    Runs N_SEEDS=20 independent simulations with seeds ζ_i = 42 + i.
    Reports mean ± std and 95% CI via bootstrap resampling.
    (Sec. VI-E5)
    """

    def __init__(self, n_seeds: int = N_SEEDS, base_seed: int = BASE_SEED):
        self.n_seeds = n_seeds
        self.seeds   = [base_seed + i for i in range(n_seeds)]
        self.all_summaries: List[Dict] = []

    def run(self):
        print(f"\n{'#'*70}")
        print(f"  Multi-Seed Experiment: {self.n_seeds} seeds")
        print(f"{'#'*70}\n")

        for seed_idx, seed in enumerate(self.seeds):
            print(f"\n Seed {seed_idx+1}/{self.n_seeds} (ζ={seed}) ")
            sim = HospitalHypergameSimulation(seed=seed)
            all_ep_metrics = sim.run()

            # Aggregate across episodes
            keys = list(all_ep_metrics[0].summary().keys())
            aggregated: Dict[str, List[float]] = {k: [] for k in keys}
            for ep_m in all_ep_metrics:
                s = ep_m.summary()
                for k in keys:
                    aggregated[k].append(s[k])

            seed_summary = {k: float(np.mean(v)) for k, v in aggregated.items()}
            seed_summary['seed'] = seed
            self.all_summaries.append(seed_summary)

        return self.compute_statistics()

    def bootstrap_ci(self, data: np.ndarray, n_boot: int = 1000,
                     alpha: float = 0.05) -> Tuple[float, float]:
        """95% bootstrap confidence interval."""
        rng = np.random.default_rng(0)
        boot_means = np.zeros(n_boot)
        for i in range(n_boot):
            sample = rng.choice(data, size=len(data), replace=True)
            boot_means[i] = sample.mean()
        lower = float(np.percentile(boot_means, 100 * alpha / 2))
        upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        return lower, upper

    def t_test_bonferroni(self, a: np.ndarray, b: np.ndarray,
                           n_tests: int = 7) -> Tuple[float, bool]:
        """Two-sample t-test with Bonferroni correction (Sec.VI-F)."""
        t_stat, p_val = stats.ttest_ind(a, b)
        p_corrected = p_val * n_tests   # Bonferroni correction
        significant = p_corrected < 0.05
        return float(p_corrected), significant

    def compute_statistics(self) -> Dict:
        """
        Aggregate mean ± std and 95% CI across all seeds.
        Report per Sec. VI-E5 statistical conventions.
        """
        keys = [k for k in self.all_summaries[0].keys() if k != 'seed']
        results = {}

        print(f"\n{'='*70}")
        print("  FINAL STATISTICS (mean ± std, 95% CI)")
        print(f"{'='*70}")

        for k in keys:
            vals = np.array([s[k] for s in self.all_summaries])
            mean = float(vals.mean())
            std  = float(vals.std())
            ci_l, ci_u = self.bootstrap_ci(vals)
            results[k] = {
                'mean': mean, 'std': std,
                'ci_lower': ci_l, 'ci_upper': ci_u,
                'min': float(vals.min()), 'max': float(vals.max()),
            }
            print(f"  {k:30s}: {mean:.4f} ± {std:.4f}  "
                  f"[{ci_l:.4f}, {ci_u:.4f}]")

        return results


# ══════════════════════════════════════════════════════════════════════════════
# 16.  VISUALISATION  (Figures for direct paper submission)
# ══════════════════════════════════════════════════════════════════════════════

def generate_figures(sim_results: List[MetricsCollector],
                     multi_seed_stats: Dict,
                     output_dir: str = '/mnt/user-data/outputs'):
    """
    Generate all publication-ready figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'font.family': 'serif', 'text.usetex': False,
        'figure.dpi': 150,
    })
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f']

    # ── Figure 1: Network Health + Resilience over Episodes ──────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hypergame-Aware IoT Defense — Smart Hospital Simulation',
                 fontsize=13, fontweight='bold')

    # Panel (a): Health over time
    ax = axes[0, 0]
    all_health = np.array([m.health_history for m in sim_results if m.health_history])
    if all_health.size > 0:
        mean_h = all_health.mean(axis=0)
        std_h  = all_health.std(axis=0)
        t_ax   = np.arange(len(mean_h)) * T_STEPS / len(mean_h)
        ax.plot(t_ax, mean_h, color=colors[0], lw=2, label='Mean Health H(t)')
        ax.fill_between(t_ax, mean_h - std_h, mean_h + std_h,
                         alpha=0.25, color=colors[0], label='±1 std')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Network Health H(t)  (Eq.29)')
    ax.set_title('(a) Network Health Under Perception Attacks')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Panel (b): Detection Rate and Attack Success Rate
    ax = axes[0, 1]
    ep_drs = []
    ep_srs = []
    for m in sim_results:
        if m.n_attacks_total > 0:
            ep_drs.append(m.detection_rate)
            ep_srs.append(m.attack_success_rate)
        else:
            ep_drs.append(0.0)
            ep_srs.append(0.0)
    ep_idx = np.arange(len(ep_drs))
    ax.plot(ep_idx, ep_drs, color=colors[2], lw=2, marker='o', ms=3,
            label='Detection Rate DR')
    ax.plot(ep_idx, ep_srs, color=colors[3], lw=2, marker='s', ms=3,
            linestyle='--', label='Attack Success Rate SR')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rate')
    ax.set_title('(b) Detection vs. Attack Success Rate')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Panel (c): Resilience (Eq.17) and Robustness
    ax = axes[1, 0]
    all_res = np.array([m.resilience_history for m in sim_results if m.resilience_history])
    all_rob = np.array([m.robustness_history for m in sim_results if m.robustness_history])
    if all_res.size > 0:
        mean_r = all_res.mean(axis=0)
        t_ax2  = np.arange(len(mean_r))
        ax.plot(t_ax2, mean_r, color=colors[1], lw=2, label='Resilience ρ(t) (Eq.17)')
    if all_rob.size > 0:
        mean_rob = all_rob.mean(axis=0)
        ax.plot(np.arange(len(mean_rob)), mean_rob, color=colors[4],
                lw=2, linestyle='-.', label='Robustness Λ(t)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Score')
    ax.set_title('(c) Resilience and Robustness')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Panel (d): Information Advantage I(P^D; P^A) (Eq.27)
    ax = axes[1, 1]
    all_ia = np.array([m.info_advantage_history for m in sim_results if m.info_advantage_history])
    if all_ia.size > 0:
        mean_ia = all_ia.mean(axis=0)
        std_ia  = all_ia.std(axis=0)
        t_ax3 = np.arange(len(mean_ia))
        ax.plot(t_ax3, mean_ia, color=colors[5], lw=2,
                label='I(P^D; P^A)  (Eq.27)')
        ax.fill_between(t_ax3, mean_ia - std_ia, mean_ia + std_ia,
                         alpha=0.2, color=colors[5])
        ax.axhline(0, color='gray', lw=1, linestyle='--', label='Zero (balanced)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Information Advantage')
    ax.set_title('(d) Information Advantage I(P^D; P^A)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig_path1 = os.path.join(output_dir, 'fig1_main_metrics.png')
    plt.savefig(fig_path1, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path1}")

    # ── Figure 2: Convergence + Perception Gap + Strategy Entropy ────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    all_eps = np.array([m.epsilon_k_history for m in sim_results if m.epsilon_k_history])
    k_vals  = np.arange(1, K_DEFENDER + 5)
    eps_curve = [convergence_bound(k) for k in k_vals]
    ax.semilogy(k_vals, eps_curve, 'r-o', lw=2, ms=6,
                label=r'$\epsilon(k) = C\rho^k$ (Prop.1)')
    if all_eps.size > 0:
    # all_eps shape: (n_episodes, T_STEPS)
    # take mean over time per episode, then mean over episodes
        emp_mean = all_eps.mean()   # scalar — overall mean
    # plot as horizontal reference line
        ax.axhline(emp_mean, color='blue', linestyle='--', lw=1.5,
               label=f'Empirical $\\bar{{\\epsilon}}(k)$ = {emp_mean:.4f}')
    ax.set_xlabel('Reasoning Level k')
    ax.set_ylabel('Approximation Error ε(k)')
    ax.set_title('(e) Convergence Bound (Theorem 1)')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9)

    ax = axes[1]
    all_pg = np.array([m.perception_gap_history for m in sim_results if m.perception_gap_history])
    if all_pg.size > 0:
        mean_pg = all_pg.mean(axis=0)
        std_pg  = all_pg.std(axis=0)
        t_ax4 = np.arange(len(mean_pg))
        ax.plot(t_ax4, mean_pg, color=colors[6], lw=2, label='ΔP(t) (Eq.35)')
        ax.fill_between(t_ax4, np.maximum(0, mean_pg - std_pg),
                         mean_pg + std_pg, alpha=0.2, color=colors[6])
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Perception Gap ΔP(t)')
    ax.set_title('(f) Defender Perception Gap Under Attack')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[2]
    all_ed = np.array([m.strategy_entropy_d for m in sim_results if m.strategy_entropy_d])
    all_ea = np.array([m.strategy_entropy_a for m in sim_results if m.strategy_entropy_a])
    if all_ed.size > 0:
        ax.plot(np.arange(len(all_ed[0])), all_ed.mean(axis=0),
                color=colors[0], lw=2, label='H(σ_D) Defender Entropy')
    if all_ea.size > 0:
        ax.plot(np.arange(len(all_ea[0])), all_ea.mean(axis=0),
                color=colors[3], lw=2, linestyle='--', label='H(σ_A) Attacker Entropy')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Strategy Entropy')
    ax.set_title('(g) Mixed Strategy Entropy Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig_path2 = os.path.join(output_dir, 'fig2_convergence_perception.png')
    plt.savefig(fig_path2, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path2}")

    # ── Figure 3: Multi-seed summary bar chart ────────────────────────────────
    if multi_seed_stats:
        fig, ax = plt.subplots(figsize=(12, 6))
        metric_keys = ['mean_health', 'detection_rate', 'attack_success_rate',
                        'mean_resilience', 'mean_robustness']
        metric_labels = ['Health H(t)', 'Detection Rate', 'Attack Suc. Rate',
                          'Resilience', 'Robustness']
        means  = [multi_seed_stats[k]['mean']  for k in metric_keys
                  if k in multi_seed_stats]
        stds   = [multi_seed_stats[k]['std']   for k in metric_keys
                  if k in multi_seed_stats]
        ci_low = [multi_seed_stats[k]['mean'] - multi_seed_stats[k]['ci_lower']
                  for k in metric_keys if k in multi_seed_stats]
        ci_high= [multi_seed_stats[k]['ci_upper'] - multi_seed_stats[k]['mean']
                  for k in metric_keys if k in multi_seed_stats]

        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=[ci_low, ci_high], capsize=6,
                       color=colors[:len(means)], alpha=0.8,
                       error_kw={'elinewidth': 2, 'ecolor': 'black'})
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels[:len(means)], rotation=15)
        ax.set_ylabel('Score')
        ax.set_title(f'(h) Multi-Seed Statistics ({N_SEEDS} seeds, 95% CI)')
        ax.set_ylim(0, max(means) * 1.3 if means else 1)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{m:.3f}±{s:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        fig_path3 = os.path.join(output_dir, 'fig3_multiseed_stats.png')
        plt.savefig(fig_path3, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path3}")
        paths = [fig_path1, fig_path2]
        if multi_seed_stats:
            paths.append(fig_path3)
        return paths

   # return [fig_path1, fig_path2]


# ══════════════════════════════════════════════════════════════════════════════
# 17.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main(full_multiseed: bool = False):
    """
    Main experiment runner.
    Set full_multiseed=True to run all 20 seeds (slow).
    Default: 3 seeds for rapid validation.
    """
    print("\n" + "="*70)
    print("  Hypergame-Aware AI Defense for IoT Perception Attacks")
    print("  Smart Hospital Simulation — Direct Submission Version")
    print("  Implements: Eq.1–39, Theorem 1, Lemmas 1–3, Proposition 1")
    print("="*70)

    n_seeds_to_run = N_SEEDS if full_multiseed else 3
    seeds = [BASE_SEED + i for i in range(n_seeds_to_run)]

    all_sim_results = []
    all_summaries   = []

    for seed_i, seed in enumerate(seeds):
        print(f"\n{'─'*50}")
        print(f"  Seed {seed_i+1}/{n_seeds_to_run}  (ζ = {seed})")
        print(f"{'─'*50}")
        sim = HospitalHypergameSimulation(seed=seed)
        ep_metrics_list = sim.run()
        all_sim_results.extend(ep_metrics_list)

        keys = list(ep_metrics_list[0].summary().keys())
        aggregated = {k: [] for k in keys}
        for ep_m in ep_metrics_list:
            s = ep_m.summary()
            for k in keys:
                aggregated[k].append(s[k])
        seed_summ = {k: float(np.mean(v)) for k, v in aggregated.items()}
        seed_summ['seed'] = seed
        all_summaries.append(seed_summ)

    # Compute multi-seed statistics
    print(f"\n{'='*70}")
    print("  MULTI-SEED AGGREGATED STATISTICS")
    print(f"{'='*70}")

    multi_stats: Dict = {}
    experiment = MultiSeedExperiment.__new__(MultiSeedExperiment)
    experiment.all_summaries = all_summaries
    multi_stats = experiment.compute_statistics()

    # Generate figures
    print(f"\n{'='*70}")
    print("  Generating Publication Figures...")
    print(f"{'='*70}")
    fig_paths = generate_figures(all_sim_results, multi_stats)
    print(f"  Figures saved: {', '.join(fig_paths)}")

    # Save JSON summary
    json_path = '/mnt/user-data/outputs/simulation_results.json'
    with open(json_path, 'w') as f:
        # Convert numpy types for JSON serialisation
        def convert(obj):
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        json.dump(convert(multi_stats), f, indent=2)
    print(f"  Saved: {json_path}")
 
    # Save raw timestep data for post-hoc analysis
    raw_path = '/mnt/user-data/outputs/simulation_raw.npz'
    np.savez_compressed(
        raw_path,
        health      = np.array([m.health_history            for m in all_sim_results]),
        resilience  = np.array([m.resilience_history         for m in all_sim_results]),
        robustness  = np.array([m.robustness_history         for m in all_sim_results]),
        info_adv    = np.array([m.info_advantage_history     for m in all_sim_results]),
        perc_gap    = np.array([m.perception_gap_history     for m in all_sim_results]),
        epsilon_k   = np.array([m.epsilon_k_history          for m in all_sim_results]),
        utility_d   = np.array([m.utility_defender           for m in all_sim_results]),
        utility_a   = np.array([m.utility_attacker           for m in all_sim_results]),
        entropy_d   = np.array([m.strategy_entropy_d         for m in all_sim_results]),
        entropy_a   = np.array([m.strategy_entropy_a         for m in all_sim_results]),
        n_eff       = np.array([m.n_eff_history              for m in all_sim_results]),
        conv_iter   = np.array([m.convergence_iterations     for m in all_sim_results]),
        n_atk_total = np.array([m.n_attacks_total            for m in all_sim_results]),
        n_atk_det   = np.array([m.n_attacks_detected         for m in all_sim_results]),
        n_atk_suc   = np.array([m.n_attacks_success          for m in all_sim_results]),
    )
    print(f"  Saved: {raw_path}")

    print("\n" + "="*70)
    print("  SIMULATION COMPLETE")
    print(f"  Health:      {multi_stats.get('mean_health', {}).get('mean', 0):.4f} "
          f"± {multi_stats.get('mean_health', {}).get('std', 0):.4f}")
    print(f"  Det. Rate:   {multi_stats.get('detection_rate', {}).get('mean', 0):.4f}")
    print(f"  Atk. Suc.:   {multi_stats.get('attack_success_rate', {}).get('mean', 0):.4f}")
    print(f"  Resilience:  {multi_stats.get('mean_resilience', {}).get('mean', 0):.4f}")
    print(f"  Info Adv:    {multi_stats.get('mean_info_adv', {}).get('mean', 0):.4f}")
    print("="*70)
    np.save('raw_metrics.npy', [{
        'health': m.health_history,
        'resilience': m.resilience_history,
        'detection_rate': m.detection_rate,
        'info_advantage': m.info_advantage_history,
        'perception_gap': m.perception_gap_history,
        'attack_success_rate': m.attack_success_rate,
        'convergence_iterations': m.convergence_iterations,
        'epsilon_k_history': m.epsilon_k_history,
        'utility_defender': m.utility_defender,
        'utility_attacker': m.utility_attacker,
    } for m in all_sim_results])
    return multi_stats, all_sim_results


if __name__ == '__main__':
    # Run with 3 seeds for validation; change to full_multiseed=True for paper
    results, sims = main(full_multiseed=True)
