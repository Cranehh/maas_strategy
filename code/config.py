"""
Central configuration for MaaS ABM optimization framework.
All model parameters, file paths, and hyperparameters.
"""
import os
import numpy as np

# ============================================================
# A. File Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, '数据')

# 2023 Travel Survey
FAMILYMEMBER_CSV = os.path.join(DATA_DIR, '居民出行数据', '2023', 'familymember_total_33169.csv')
FAMILY_CSV = os.path.join(DATA_DIR, '居民出行数据', '2023', 'family_total_33169.csv')
MIDTABLE_CSV = os.path.join(DATA_DIR, '居民出行数据', '2023', 'midtable_total_33169.csv')
FAMILY_AUTO_CSV = os.path.join(DATA_DIR, '居民出行数据', '2023', 'familyautomobile_total_33169.csv')

# SP Survey Data
PEOPLE_ATTITUDE_CSV = os.path.join(DATA_DIR, 'MaaS被调查者聚类加个人属性加态度.csv')
BUNDLE_DATA_CSV = os.path.join(DATA_DIR, 'question_bundle_data.csv')
FACTOR_SCORES_CSV = os.path.join(DATA_DIR, 'factor_scores.csv')
SINGLE_TRIP_XLSX = os.path.join(DATA_DIR, '单次出行数据.xlsx')

# GIS
TAZ_SHP = os.path.join(DATA_DIR, '北京市交通小区', '北京市交通小区.shp')
DISTRICT_XLSX = os.path.join(DATA_DIR, '居民出行数据', '城区信息表.xlsx')
ZONE_XLSX = os.path.join(DATA_DIR, '居民出行数据', '小区信息表.xlsx')

# Pickle
CH3_PICKLE = os.path.join(BASE_DIR, '已有代码', 'MaaS 附加值相关',
                          'HCM~nofactor5 去掉套餐中的ebiike.pickle')

# ============================================================
# B. Ch3 ICLV Parameters (extracted from HTML report)
# ============================================================
CH3_PARAMS = {
    # --- ASC (Alternative Specific Constants) ---
    'ASC_1': 2.29,    # BF (Basic Flexibility)
    'ASC_2': 2.13,    # MA (Mobility Advantage)
    'ASC_3': 1.16,    # VT (Value Traveler)
    'ASC_4': 1.38,    # UA (Unlimited Access)
    'ASC_5': 0.0,     # No-purchase (reference, fixed to 0)

    # --- Bundle attributes ---
    'B_BUN_TAXI': -0.289,
    'B_BUN_PRICERATIO': -0.325,
    'B_BUN_PRICE': -0.0475,

    # --- Travel behavior ---
    'B_BUS_NUM': 0.244,
    'B_METRO_NUM': 0.337,
    'B_TAXI_NUM': 0.464,
    'B_EBIKE_HOME': 0.297,
    'B_TRAVEL_DISTANCE_WORK': 0.034,
    'B_TRAVEL_DISTANCE_END': 0.0557,
    'B_COMBINE_SHAREBIKE': 0.085,
    'B_CAR': 0.336,
    'B_COST': 0.0406,

    # --- Socio-demographics ---
    'B_SEX': -0.191,
    'B_OCCUPY': -0.0337,
    'B_AGE3': 0.312,
    'B_AGE4': 0.833,
    'B_INCOME1': -0.0895,
    'B_INCOME2': 0.353,
    'B_EDUCATION': 0.263,
    'B_HAVECAR': 0.0859,
    'B_LICENSE': 0.241,

    # --- Latent variables (FACTOR coefficients in utility) ---
    'B_FACTOR1': 2.91,
    'B_FACTOR2': 2.05,
    'B_FACTOR3': 4.08,
    'B_FACTOR4': 0.163,
    'B_FACTOR6': -2.7,

    # --- Nesting parameter ---
    'MU1': 2.13,

    # --- Structural equation coefficients (FACTOR1) ---
    'coef1_age1': 1.6,    'coef1_age2': 1.49,   'coef1_age3': 1.47,
    'coef1_job': -0.0694,
    'coef1_income1': 1.95, 'coef1_income2': 1.89, 'coef1_income3': 1.76,
    'coef1_travel_num': 0.0672,
    'coef1_travel_distance_day': 0.0282,
    'coef1_travel_aim': 0.0178,
    'coef1_bus': -0.00129, 'coef1_metro': 0.0533,
    'coef1_taxi': -0.0381, 'coef1_ebike': -0.0737,
    'coef1_bike': -0.00259,
    'sigma_s1': 0.93,

    # --- Structural equation coefficients (FACTOR2) ---
    'coef2_age1': 1.08,   'coef2_age2': 0.984,  'coef2_age3': 0.868,
    'coef2_job': -0.12,
    'coef2_income1': 1.15, 'coef2_income2': 1.15, 'coef2_income3': 1.12,
    'coef2_travel_num': 0.0648,
    'coef2_travel_distance_day': -0.0107,
    'coef2_travel_aim': 0.0297,
    'coef2_bus': 0.0336,   'coef2_metro': 0.042,
    'coef2_taxi': 0.00437, 'coef2_ebike': 0.0214,
    'coef2_bike': -0.0123,
    'sigma_s2': 0.803,

    # --- Structural equation coefficients (FACTOR3) ---
    'coef3_age1': 1.1,    'coef3_age2': 1.03,   'coef3_age3': 1.11,
    'coef3_job': -0.0389,
    'coef3_income1': 1.41, 'coef3_income2': 1.34, 'coef3_income3': 1.36,
    'coef3_travel_num': 0.0429,
    'coef3_travel_distance_day': 0.0374,
    'coef3_travel_aim': 0.0399,
    'coef3_bus': 0.00759,  'coef3_metro': 0.0151,
    'coef3_taxi': -0.0363, 'coef3_ebike': -0.0351,
    'coef3_bike': 0.0123,
    'sigma_s3': 0.691,

    # --- Structural equation coefficients (FACTOR4) ---
    'coef4_age1': 0.805,  'coef4_age2': 0.855,  'coef4_age3': 0.882,
    'coef4_job': 0.0182,
    'coef4_income1': 1.27, 'coef4_income2': 1.15, 'coef4_income3': 0.986,
    'coef4_travel_num': -0.0191,
    'coef4_travel_distance_day': 0.0239,
    'coef4_travel_aim': 0.0817,
    'coef4_bus': 0.0641,   'coef4_metro': 0.0134,
    'coef4_taxi': 0.0119,  'coef4_ebike': 0.0175,
    'coef4_bike': 0.00226,
    'sigma_s4': 0.681,

    # --- Structural equation coefficients (FACTOR6) ---
    'coef6_age1': 1.07,   'coef6_age2': 0.983,  'coef6_age3': 0.98,
    'coef6_job': -0.0526,
    'coef6_income1': 1.24, 'coef6_income2': 1.29, 'coef6_income3': 1.27,
    'coef6_travel_num': 0.0842,
    'coef6_travel_distance_day': 0.0295,
    'coef6_travel_aim': 0.0429,
    'coef6_bus': 0.0403,   'coef6_metro': 0.0168,
    'coef6_taxi': 0.0176,  'coef6_ebike': -0.00292,
    'coef6_bike': 0.0436,
    'sigma_s6': 0.677,

    # --- Measurement equation intercepts ---
    'INTER_at9': -0.0184,  'INTER_at10': 0.00806,
    'INTER_at11': -0.198,   'INTER_at13': -0.157,
    'INTER_at14': -0.599,   'INTER_at17': -0.38,
    'INTER_at22': -0.0475,  'INTER_at23': -0.21,
    'INTER_at24': -0.0767,  'INTER_at25': -0.126,
    'INTER_at19': -0.22,    'INTER_at15': -0.0634,
    'INTER_at4': -0.468,    'INTER_at6': -0.0883,

    # --- Measurement equation factor loadings ---
    'B_at9': 0.924,   'B_at10': 0.803,  'B_at11': 1.02,
    'B_at13': 0.991,  'B_at14': 1.1,    'B_at17': 1.08,
    'B_at22': 1.11,   'B_at23': 1.04,   'B_at24': 1.03,   'B_at25': 1.0,
    'B_at19': 1.05,   'B_at15': 1.07,
    'B_at4': 1.1,     'B_at6': 0.87,

    # --- Measurement equation sigma stars ---
    'SIGMA_STAR_at9': 1.13,   'SIGMA_STAR_at10': 1.0,
    'SIGMA_STAR_at11': 1.0,   'SIGMA_STAR_at13': 1.14,
    'SIGMA_STAR_at14': 1.05,  'SIGMA_STAR_at17': 1.22,
    'SIGMA_STAR_at22': 1.0,   'SIGMA_STAR_at23': 0.922,
    'SIGMA_STAR_at24': 0.956, 'SIGMA_STAR_at25': 1.06,
    'SIGMA_STAR_at19': 0.888, 'SIGMA_STAR_at15': 0.95,
    'SIGMA_STAR_at4': 1.22,   'SIGMA_STAR_at6': 0.953,

    # --- Ordinal thresholds ---
    'delta_1p': 1.03,  'delta_2p': 1.13,  'delta_3p': 1.22,
}

# ============================================================
# C. Ch1 LC-HCM Parameters
#    Calibrated from Ch1 source code (LC-HCM~去掉at13，保留1和5的距离5-最终版.py).
#    FACTOR1 structural equation coefficients adapted from Ch3 estimates
#    (same functional form, same survey population).
#    Choice model and class-membership coefficients calibrated to produce
#    realistic trial probability distribution (target mean ~0.3-0.5 monthly).
#    No separate Ch1 pickle/HTML results file exists.
# ============================================================
CH1_PARAMS = {
    # Class membership (logit for class 1=MaaS-receptive vs class 2=reference)
    # Source: utilityClass1 = exp(C_ASC + C_B_MAASFAMILAR*MaasFamiliar
    #   + C_B_TAXI*Al_taxi + C_B_PT*Al_PT + C_B_BIKE*Al_bike
    #   + C_B_HAVECAR*Carown + C_B_TRAVEL_DISTANCE_END*DistanceWeekend
    #   + C_B_FACTOR1*FACTOR1)
    'C_ASC': -0.8,
    'C_B_MAASFAMILAR': 0.6,
    'C_B_TAXI': 0.25,
    'C_B_PT': 0.35,
    'C_B_BIKE': 0.15,
    'C_B_HAVECAR': -0.5,
    'C_B_TRAVEL_DISTANCE_END': 0.2,
    'C_B_FACTOR1': 0.3,

    # Class 1 utility coefficients (MaaS-receptive class)
    # Source code structure: V11, V41, V51, V61, V71 (5 alts: no-transfer, M1-M4)
    # ASC_11=0 (fixed), ASC_41..71 are negative (inertia favors no-transfer)
    # Calibrated so P_single_trip ≈ 0.01-0.02 → P_monthly ≈ 0.3-0.5 with ~47 trips
    'ASC_41': -4.8,  'ASC_51': -4.6,  'ASC_61': -4.3,  'ASC_71': -4.1,
    'B_RAIL_TIME1': -0.015,
    'B_TRIP_TIME1': -0.012,
    'B_FIRST_PT1': 0.45,
    'B_FIRST_TAXI1': 0.35,
    'B_SAME_CHOICE_CAR1': 1.2,
    'B_SAME_CHOICE_TAXI1': 0.6,
    'B_DISTANCE51': 0.3,
    'B_DEPARTTIME31': 0.15,

    # Class 2 utility coefficients (MaaS-resistant class)
    # Stronger negative ASCs, weaker mode-shift incentives
    'ASC_42': -6.0,  'ASC_52': -5.8,  'ASC_62': -5.5,  'ASC_72': -5.3,
    'B_RAIL_TIME2': -0.008,
    'B_TRIP_TIME2': -0.006,
    'B_FIRST_PT2': 0.25,
    'B_FIRST_TAXI2': 0.15,
    'B_SAME_CHOICE_CAR2': 1.8,
    'B_SAME_CHOICE_TAXI2': 0.5,
    'B_DISTANCE52': 0.15,
    'B_DEPARTTIME32': 0.08,

    # Ch1 FACTOR1 structural equation
    # Adapted from Ch3 FACTOR1 estimates (same form, same variables)
    # Source active terms: gender, age1-3, job, income1-2, education,
    #   travel_num, travel_distance_day, 6d-6g, car_home, metro, ebike
    'ch1_coef1_gender': 0.12,
    'ch1_coef1_age1': 1.45, 'ch1_coef1_age2': 1.35, 'ch1_coef1_age3': 1.30,
    'ch1_coef1_job': -0.06,
    'ch1_coef1_income1': 1.80, 'ch1_coef1_income2': 1.75,
    'ch1_coef1_education': 0.15,
    'ch1_coef1_travel_num': 0.06,
    'ch1_coef1_travel_distance_day': 0.025,
    'ch1_coef1_6d': 0.07, 'ch1_coef1_6e': 0.05,
    'ch1_coef1_6f': 0.04, 'ch1_coef1_6g': 0.03,
    'ch1_coef1_car_home': -0.12,
    'ch1_coef1_metro': 0.05,
    'ch1_coef1_ebike': -0.06,
    'ch1_sigma_s1': 0.85,
}

# ============================================================
# D. Strategy Parameter Bounds (17-dim theta)
# ============================================================
# theta layout:
# [0]  taxi_BF:       taxi trips in BF bundle [0, 350]
# [1]  ps_BF:         price scale for BF [0.7, 1.3]
# [2]  taxi_MA:       taxi trips in MA bundle [0, 520]
# [3]  ps_MA:         price scale for MA [0.7, 1.3]
# [4]  ps_VT:         price scale for VT [0.7, 1.3]
# [5]  ps_UA:         price scale for UA [0.7, 1.3]
# [6]  tau_high:      awareness threshold high [0.3, 0.8]
# [7]  tau_low:       awareness threshold low [0.1, 0.5]
# [8]  delta_up:      marketing budget increase rate [0.0, 0.5]
# [9]  delta_down:    marketing budget decrease rate [0.0, 0.5]
# [10] freq_adj:      adjustment frequency (quarters) [1, 4]
# [11] B_total:       total marketing budget (万元/月) [50, 500]
# [12] gamma_potential: marketing allocation to high-potential areas [0.3, 0.9]
# [13] gamma_gap:     marketing allocation to awareness-gap areas [0.1, 0.7]
# [14] c_conc:        concentration parameter for spatial targeting [0.1, 0.9]
# [15] time_improvement: MaaS trip time improvement ratio [0.0, 0.3]
# [16] price_discount:   MaaS trip price discount ratio [0.0, 0.3]

THETA_NAMES = [
    'taxi_BF', 'ps_BF', 'taxi_MA', 'ps_MA', 'ps_VT', 'ps_UA',
    'tau_high', 'tau_low', 'delta_up', 'delta_down', 'freq_adj',
    'B_total', 'gamma_potential', 'gamma_gap', 'c_conc',
    'time_improvement', 'price_discount',
]

THETA_LOWER = np.array([
    0.0, 0.7, 0.0, 0.7, 0.7, 0.7,
    0.3, 0.1, 0.0, 0.0, 1.0,
    50.0, 0.3, 0.1, 0.1,
    0.0, 0.0,
], dtype=np.float64)

THETA_UPPER = np.array([
    350.0, 1.3, 520.0, 1.3, 1.3, 1.3,
    0.8, 0.5, 0.5, 0.5, 4.0,
    500.0, 0.9, 0.7, 0.9,
    0.3, 0.3,
], dtype=np.float64)

N_THETA = len(THETA_NAMES)  # 17

# Default theta (baseline scenario)
THETA_DEFAULT = np.array([
    0.0, 1.0, 350.0, 1.0, 1.0, 1.0,
    0.5, 0.2, 0.1, 0.1, 2.0,
    200.0, 0.6, 0.3, 0.5,
    0.1, 0.1,
], dtype=np.float64)

# ============================================================
# E. ABM Hyperparameters
# ============================================================
N_WEEKS = 156           # 3 years
P_INNOV = 0.02          # Bass innovation coefficient (weekly)
P_IMIT = 0.05           # Bass imitation coefficient (weekly)
BETA_LOCAL = 0.8        # Local (TAZ) influence weight
BETA_REMOTE = 0.2       # City-wide influence weight
AWARENESS_THRESHOLD = 0.3  # Threshold for aware->trial eligible
SATISFACTION_ALPHA = 0.7   # Exponential smoothing for satisfaction
SATISFACTION_NEW_WEIGHT = 0.3  # Weight for new satisfaction signal
CHURN_THRESHOLD = 0.3      # Satisfaction below which churn risk starts
CHURN_CONSECUTIVE_WEEKS = 4  # Weeks below threshold before churn
COOLDOWN_WEEKS = 8           # Weeks before churned can re-enter
MIN_TRIALS_FOR_SUBSCRIBE = 2  # Minimum trials before subscription eligible

# Agent status codes
STATUS_UNAWARE = 0
STATUS_AWARE = 1
STATUS_TRIAL = 2
STATUS_SUBSCRIBER = 3
STATUS_CHURNED = 4

# Population weight (Beijing ~20M / 79K agents ≈ 25)
AGENT_WEIGHT = 25.0

# ============================================================
# F. Bundle Base Prices (元/月)
# ============================================================
PRICE_BASE_BF = 80.0       # Basic Flexibility
PRICE_BASE_MA = 230.0      # Mobility Advantage
PRICE_BASE_VT = 960.0      # Value Traveler
PRICE_BASE_UA = 1556.8     # Unlimited Access

# ============================================================
# G. Carbon Emission Factors
# ============================================================
CAR_CO2_PER_KM = 0.196     # kg CO2 per km for private car
PT_CO2_PER_KM = 0.035      # kg CO2 per km for public transit
TAXI_CO2_PER_KM = 0.196    # kg CO2 per km for taxi/ride-hail

# ============================================================
# H. Optimization Settings
# ============================================================
LHS_SAMPLES = 100          # Phase A initial samples
NSGA2_POP_SIZE = 200       # NSGA-II population size
NSGA2_N_GEN = 300          # Number of generations
INFILL_PER_GEN = 10        # ABM evaluations per generation for GP update
PARETO_VERIFY_N = 50       # Number of Pareto solutions to verify
PARETO_VERIFY_SEEDS = 5    # Seeds per Pareto solution for robustness

# Objective indices
OBJ_ADOPTION = 0           # Maximize adoption rate (negated for minimization)
OBJ_REVENUE = 1            # Maximize net revenue (negated)
OBJ_EQUITY = 2             # Minimize Gini coefficient of adoption across districts
OBJ_CARBON = 3             # Maximize carbon reduction (negated)
