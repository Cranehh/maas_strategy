"""
Data loader: Load 2023 Beijing travel survey, map variables to model space,
compute factor scores, and create the 79K agent population for ABM.
"""
import numpy as np
import pandas as pd
import os

from config import (
    FAMILYMEMBER_CSV, FAMILY_CSV, MIDTABLE_CSV, FAMILY_AUTO_CSV,
    PEOPLE_ATTITUDE_CSV, FACTOR_SCORES_CSV,
    CH3_PARAMS, AGENT_WEIGHT,
)
import ch3_model


def load_familymember(path=None):
    """Load familymember_total_33169.csv (79K individuals)."""
    if path is None:
        path = FAMILYMEMBER_CSV
    df = pd.read_csv(path, encoding='utf-8-sig')
    return df


def load_family(path=None):
    """Load family_total_33169.csv (household data)."""
    if path is None:
        path = FAMILY_CSV
    df = pd.read_csv(path, encoding='utf-8-sig')
    return df


def load_midtable(path=None):
    """Load midtable_total_33169.csv (169K trip records)."""
    if path is None:
        path = MIDTABLE_CSV
    df = pd.read_csv(path, encoding='utf-8-sig')
    return df


def aggregate_trips_per_person(midtable):
    """Aggregate trip records to per-person monthly travel features.

    Returns DataFrame indexed by (家庭编号, 成员编号) with columns:
    - trips_per_month, bus_trips, metro_trips, taxi_trips, bike_trips, ebike_trips, car_trips
    - avg_distance, avg_cost, car_km_month
    """
    mid = midtable.copy()

    # Mode codes from 交通方式的编号 (ModelMode)
    # 1=walk, 2=bike, 3=ebike, 4=car_driver, 5=car_passenger, 6=taxi,
    # 7=bus, 8=metro, 9=other
    mode_col = 'ModelMode' if 'ModelMode' in mid.columns else '交通方式的编号'
    dist_col = '单次出行的距离'
    cost_col = '出行费用之金额'

    # Create person ID from 家庭编号 and 家庭成员编号
    mid['person_id'] = mid['家庭编号'].astype(str) + '_' + mid['家庭成员编号'].astype(str)

    agg = mid.groupby('person_id').agg(
        trip_count=(mode_col, 'count'),
        bus_trips=(mode_col, lambda x: ((x == 7) | (x == 8)).sum()),  # bus+metro combined for freq
        metro_trips=(mode_col, lambda x: (x == 8).sum()),
        taxi_trips=(mode_col, lambda x: (x == 6).sum()),
        bike_trips=(mode_col, lambda x: (x == 2).sum()),
        ebike_trips=(mode_col, lambda x: (x == 3).sum()),
        car_trips=(mode_col, lambda x: ((x == 4) | (x == 5)).sum()),
        avg_distance=(dist_col, 'mean'),
        total_cost=(cost_col, 'sum'),
        car_distance=(dist_col, lambda x: x[mid.loc[x.index, mode_col].isin([4, 5])].sum()
                      if hasattr(x, 'index') else 0),
    ).reset_index()

    # Convert daily to monthly (survey is typically one day, multiply by 22 workdays)
    agg['trips_per_month'] = agg['trip_count'] * 22
    agg['car_km_month'] = agg['car_trips'] * agg['avg_distance'] * 22

    return agg


def aggregate_trips_simple(midtable):
    """Simplified trip aggregation that's more robust to column name issues."""
    mid = midtable.copy()

    # Find the mode column
    mode_candidates = ['ModelMode', '交通方式的编号']
    mode_col = None
    for c in mode_candidates:
        if c in mid.columns:
            mode_col = c
            break

    dist_col = '单次出行的距离'
    cost_col = '出行费用之金额'

    # Build person_id
    mid['person_id'] = mid['家庭编号'].astype(str) + '_' + mid['家庭成员编号'].astype(str)

    if mode_col is None:
        # Fallback: just count trips
        agg = mid.groupby('person_id').size().reset_index(name='trip_count')
        agg['bus_trips'] = 0
        agg['metro_trips'] = 0
        agg['taxi_trips'] = 0
        agg['bike_trips'] = 0
        agg['ebike_trips'] = 0
        agg['car_trips'] = 0
        agg['avg_distance'] = 10.0
        agg['total_cost'] = 100.0
        agg['car_km_month'] = 0.0
        agg['trips_per_month'] = agg['trip_count'] * 22
        return agg

    # Ensure mode_col is numeric
    mid[mode_col] = pd.to_numeric(mid[mode_col], errors='coerce').fillna(0).astype(int)

    # Per-person aggregation
    groups = mid.groupby('person_id')

    agg = pd.DataFrame({
        'person_id': list(groups.groups.keys()),
        'trip_count': groups.size().values,
    })

    mode_data = mid.groupby('person_id')[mode_col].apply(list).to_dict()
    dist_data = mid.groupby('person_id')[dist_col].apply(list).to_dict() if dist_col in mid.columns else {}
    cost_data = mid.groupby('person_id')[cost_col].apply(list).to_dict() if cost_col in mid.columns else {}

    bus_trips = []
    metro_trips = []
    taxi_trips = []
    bike_trips = []
    ebike_trips = []
    car_trips = []
    avg_dists = []
    total_costs = []
    car_dists = []

    for pid in agg['person_id']:
        modes = mode_data.get(pid, [])
        dists = dist_data.get(pid, [0])
        costs = cost_data.get(pid, [0])

        bus_trips.append(sum(1 for m in modes if m == 7))
        metro_trips.append(sum(1 for m in modes if m == 8))
        taxi_trips.append(sum(1 for m in modes if m == 6))
        bike_trips.append(sum(1 for m in modes if m == 2))
        ebike_trips.append(sum(1 for m in modes if m == 3))
        car_trips.append(sum(1 for m in modes if m in (4, 5)))

        valid_dists = [d for d in dists if isinstance(d, (int, float)) and d > 0]
        avg_dists.append(np.mean(valid_dists) if valid_dists else 10.0)

        valid_costs = [c for c in costs if isinstance(c, (int, float)) and c >= 0]
        total_costs.append(sum(valid_costs))

        car_d = sum(d for d, m in zip(dists, modes)
                    if m in (4, 5) and isinstance(d, (int, float)) and d > 0)
        car_dists.append(car_d)

    agg['bus_trips'] = bus_trips
    agg['metro_trips'] = metro_trips
    agg['taxi_trips'] = taxi_trips
    agg['bike_trips'] = bike_trips
    agg['ebike_trips'] = ebike_trips
    agg['car_trips'] = car_trips
    agg['avg_distance'] = avg_dists
    agg['total_cost'] = total_costs
    agg['car_km_month'] = [ct * ad * 22 for ct, ad in zip(car_trips, avg_dists)]
    agg['trips_per_month'] = agg['trip_count'] * 22

    return agg


def map_survey_to_model_variables(member_df, family_df, trip_agg):
    """Map 2023 survey variables to Ch3 model variable encoding.

    Applies the EXACT SAME encoding as the Ch3 source code (lines 60-95):
    - age: [1-7] → [1,1,2,2,3,3,4] → dummies age1-4
    - income: [1-8] → [1,1,1,2,2,2,2,3] → dummies income1-3
    - education: [1-6] → [1,1,1,0,0,0] (本科以上=0)
    - occupy: [1-11] → [0,0,0,0,0,0,0,1,1,1,1]
    - sex: 1→0, 2→1 (survey: 1=male,2=female → model: 0=male,1=female)
    - week_bus/metro/taxi: binarize by frequency
    """
    df = member_df.copy()

    # Create person_id matching midtable
    df['person_id'] = df['家庭编号'].astype(str) + '_' + df['成员编号'].astype(str)

    # Merge with family data for income and car ownership
    fam = family_df[['家庭编号', '家庭年收入', '机动车数量', '电动自行车数量', '小区编号']].copy()
    fam = fam.rename(columns={
        '家庭年收入': 'income_home',
        '机动车数量': 'car_count',
        '电动自行车数量': 'ebike_count',
        '小区编号': 'taz_from_family',
    })
    # Take first row per family
    fam = fam.drop_duplicates(subset='家庭编号', keep='first')
    df = df.merge(fam, on='家庭编号', how='left')

    # Merge with trip aggregation
    if trip_agg is not None:
        df = df.merge(trip_agg, on='person_id', how='left')
    else:
        # Default trip values if no trip data
        for col in ['trip_count', 'bus_trips', 'metro_trips', 'taxi_trips',
                     'bike_trips', 'ebike_trips', 'car_trips', 'avg_distance',
                     'total_cost', 'car_km_month', 'trips_per_month']:
            df[col] = 0

    # Fill missing values
    df = df.fillna(0)

    # ---- Sex ----
    # Survey: 1=male, 2=female → Model: 0=male, 1=female
    df['sex'] = (df['性别'] - 1).clip(0, 1)

    # ---- Age ----
    # Calculate age from birth year (survey year 2023)
    birth_year = pd.to_numeric(df['出生年份'], errors='coerce').fillna(1980)
    age_val = 2023 - birth_year
    # Map to age categories: [1-7] bins matching SP survey encoding
    # 1: <18, 2: 18-24, 3: 25-34, 4: 35-44, 5: 45-54, 6: 55-64, 7: 65+
    age_cat = pd.cut(age_val, bins=[0, 18, 24, 34, 44, 54, 64, 200],
                     labels=[1, 2, 3, 4, 5, 6, 7]).astype(float).fillna(4)
    # Map to Ch3 encoding: [1,2,3,4,5,6,7] → [1,1,2,2,3,3,4]
    age_map = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4}
    age_grouped = age_cat.map(age_map).fillna(2)
    df['age1'] = (age_grouped == 1).astype(np.float32)
    df['age2'] = (age_grouped == 2).astype(np.float32)
    df['age3'] = (age_grouped == 3).astype(np.float32)
    df['age4'] = (age_grouped == 4).astype(np.float32)

    # ---- Income ----
    # 2023 survey 家庭年收入 uses letter codes: A=<2万, B=2-4万, C=4-8万,
    # D=8-15万, E=15-25万, F=25-40万, G=40-70万, I=70-100万, J=100-150万, K=>150万
    # Map to Ch3 income groups: 1=low(<8万), 2=middle(8-30万), 3=high(>30万)
    _income_letter_map = {
        'A': 1, 'B': 1, 'C': 1,   # <8万 → low
        'D': 2, 'E': 2,            # 8-25万 → middle
        'F': 3, 'G': 3,            # 25-70万 → high
        'I': 3, 'J': 3, 'K': 3,   # >70万 → high
    }
    income_raw = df['income_home'].astype(str).str.strip()
    # Try letter mapping first; fall back to numeric for edge cases
    income_grouped = income_raw.map(_income_letter_map)
    # For any unmapped values, try numeric conversion
    unmapped = income_grouped.isna()
    if unmapped.any():
        numeric_vals = pd.to_numeric(income_raw[unmapped], errors='coerce').fillna(3)
        numeric_map = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3}
        income_grouped[unmapped] = numeric_vals.clip(1, 7).map(numeric_map)
    income_grouped = income_grouped.fillna(2)
    df['income1'] = (income_grouped == 1).astype(np.float32)
    df['income2'] = (income_grouped == 2).astype(np.float32)
    df['income3'] = (income_grouped == 3).astype(np.float32)

    # ---- Education ----
    # Survey: 1=小学以下,2=初中,3=高中,4=大专,5=本科,6=研究生及以上
    # Ch3: [1,2,3,4,5,6] → [1,1,1,0,0,0] (本科以上=0, 意为高学历)
    edu_raw = pd.to_numeric(df['最高学历'], errors='coerce').fillna(3)
    edu_map = {1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0}
    df['education'] = edu_raw.clip(1, 6).map(edu_map).fillna(1).astype(np.float32)

    # ---- Occupation ----
    # Survey: [1-11], Ch3: [1-7]=0, [8-11]=1 (出行少的工作=0)
    occupy_raw = pd.to_numeric(df['职业'], errors='coerce').fillna(1)
    df['occupy'] = (occupy_raw >= 8).astype(np.float32)

    # ---- Driver's license ----
    # Survey: 1=有,2=无 → Model: 1=有,0=无
    license_raw = pd.to_numeric(df['是否有驾照'], errors='coerce').fillna(2)
    df['license'] = (license_raw == 1).astype(np.float32)

    # ---- Car ownership ----
    car_count = pd.to_numeric(df['car_count'], errors='coerce').fillna(0)
    df['have_car'] = (car_count > 0).astype(np.float32)

    # ---- E-bike ownership ----
    ebike_count = pd.to_numeric(df['ebike_count'], errors='coerce').fillna(0)
    df['e_bike'] = (ebike_count > 0).astype(np.float32)

    # ---- Weekly mode usage (binarize: frequent=1) ----
    # Trips per day → weekly estimate: multiply by survey_days_per_week factor
    # Threshold: ≥2 trips/day → frequent user
    df['week_bus'] = (df['bus_trips'] >= 1).astype(np.float32)
    df['week_metro'] = (df['metro_trips'] >= 1).astype(np.float32)
    df['week_taxi'] = (df['taxi_trips'] >= 1).astype(np.float32)
    df['week_ebike'] = (df['ebike_trips'] >= 1).astype(np.float32)
    df['week_bike'] = (df['bike_trips'] >= 1).astype(np.float32)

    # ---- Travel behavior ----
    # travel_num: binary (frequent traveler = many trips)
    df['travel_num'] = (df['trip_count'] >= 3).astype(np.float32)
    # travel_distance_work: binary (long commute)
    df['travel_distance_work'] = (df['avg_distance'] >= 10).astype(np.float32)
    # travel_distance_weekend: proxy (assume similar to workday for now)
    df['travel_distance_weekend'] = (df['avg_distance'] >= 15).astype(np.float32)
    # travel_aim: binary (0=commute primary, 1=diverse purposes)
    df['travel_aim'] = 0.0  # Default: commute-focused

    # ---- Cost ----
    # Monthly transport cost category: <150=0, ≥150=1
    monthly_cost = df['total_cost'] * 22  # Scale daily to monthly
    df['cost'] = (monthly_cost >= 150).astype(np.float32)

    # ---- TAZ ----
    taz_col = '家庭小区编号的环路代码' if '家庭小区编号的环路代码' in df.columns else 'taz_from_family'
    taz_raw = pd.to_numeric(df.get(taz_col, pd.Series(dtype=float)), errors='coerce').fillna(0)
    # Use family's 小区编号 as TAZ if available
    taz_family = pd.to_numeric(df.get('taz_from_family', pd.Series(dtype=float)), errors='coerce').fillna(0)
    df['taz'] = taz_family.astype(int)

    # ---- District ----
    # Derive district from 家庭所属城区 (1-16 for Beijing districts) or 所属城区
    district_col = None
    for col_name in ['家庭所属城区', '所属城区']:
        if col_name in df.columns:
            district_col = col_name
            break
    if district_col is not None:
        df['district'] = pd.to_numeric(df[district_col], errors='coerce').fillna(0).astype(int)
    else:
        # Fallback: derive from TAZ code (first 2 digits often encode district)
        df['district'] = (df['taz'] // 1000).clip(0, 16).astype(int)

    # ---- Proxy variables for Ch1 (not in 2023 survey) ----
    # MaaS familiarity: assume 30% of population
    df['MaasFamiliar'] = 0.0
    np.random.seed(42)
    familiar_mask = np.random.random(len(df)) < 0.3
    df.loc[familiar_mask, 'MaasFamiliar'] = 1.0

    # First mode choice proxy
    df['first_car'] = df['have_car'].copy()
    df['first_taxi'] = (df['week_taxi'] == 1).astype(np.float32)
    df['first_pt'] = ((df['week_bus'] == 1) | (df['week_metro'] == 1)).astype(np.float32)
    # Normalize: only one first mode per person
    first_sum = df['first_car'] + df['first_taxi'] + df['first_pt']
    first_sum = first_sum.clip(lower=1)  # Avoid division by zero
    df['first_car'] = df['first_car'] / first_sum
    df['first_taxi'] = df['first_taxi'] / first_sum
    df['first_pt'] = df['first_pt'] / first_sum

    # Distance5: long weekend travel distance
    df['distance5'] = (df['avg_distance'] >= 30).astype(np.float32)
    # Normal departure time
    df['normal_depart'] = 0.5

    # Additional attitude variables (proxy with defaults)
    df['Al_taxi'] = df['week_taxi'].copy()
    df['Al_PT'] = ((df['week_bus'] == 1) | (df['week_metro'] == 1)).astype(np.float32)
    df['Al_bike'] = df['week_bike'].copy()
    df['Al_sharedbike'] = 0.0
    df['Carown'] = df['have_car'].copy()

    # c6 (car attitude), c7 (combine share bike)
    df['c6'] = df['have_car'].copy()
    df['c7'] = 0.0

    # Cost a-la-carte (monthly transport spending without subscription)
    df['cost_alacarte'] = monthly_cost.clip(lower=50)

    # Car km per month
    df['car_km_month'] = pd.to_numeric(df['car_km_month'], errors='coerce').fillna(0)

    return df


def compute_and_validate_factors(agents_df, params=None):
    """Compute factor scores and optionally validate against factor_scores.csv."""
    if params is None:
        params = CH3_PARAMS

    # Build a temporary agents dict for factor computation
    agents_dict = {}
    for col in ['age1', 'age2', 'age3', 'age4', 'occupy', 'income1', 'income2',
                'income3', 'travel_num', 'travel_distance_work', 'travel_aim',
                'week_bus', 'week_metro', 'week_taxi', 'week_ebike', 'week_bike']:
        agents_dict[col] = agents_df[col].values.astype(np.float32)

    factors = ch3_model.compute_factor_scores(agents_dict, params)

    for k, v in factors.items():
        agents_df[k] = v

    return agents_df


def create_agent_population():
    """Main function: Load 2023 survey and create agent dict for ABM.

    Returns:
        agents: dict[str, np.ndarray] with all model variables for 79K agents
    """
    print("  Loading survey data...")
    member_df = load_familymember()
    family_df = load_family()

    print("  Loading trip records...")
    try:
        midtable = load_midtable()
        print(f"  Aggregating {len(midtable)} trips...")
        trip_agg = aggregate_trips_simple(midtable)
    except Exception as e:
        print(f"  Warning: Could not load midtable: {e}")
        print("  Using default trip values.")
        trip_agg = None

    print("  Mapping variables...")
    df = map_survey_to_model_variables(member_df, family_df, trip_agg)

    print("  Computing factor scores...")
    df = compute_and_validate_factors(df)

    # Validate factor scores against reference (if available)
    if os.path.exists(FACTOR_SCORES_CSV):
        try:
            ref = pd.read_csv(FACTOR_SCORES_CSV)
            print(f"  Reference factor_scores.csv has {len(ref)} rows")
            # Note: reference uses SP survey people (1260), not 2023 survey
            # So validation is qualitative (similar range/distribution)
            for f_name in ['FACTOR1', 'FACTOR2', 'FACTOR3', 'FACTOR4', 'FACTOR6']:
                f_lower = f_name.lower()
                if f_lower in df.columns and f_name in ref.columns:
                    print(f"    {f_name}: survey mean={df[f_lower].mean():.3f}, "
                          f"ref mean={ref[f_name].mean():.3f}")
        except Exception as e:
            print(f"  Warning: Could not validate factors: {e}")

    # Build agents dict
    agent_cols = [
        'sex', 'age1', 'age2', 'age3', 'age4',
        'income1', 'income2', 'income3',
        'education', 'occupy', 'license', 'have_car', 'e_bike',
        'week_bus', 'week_metro', 'week_taxi', 'week_ebike', 'week_bike',
        'travel_num', 'travel_distance_work', 'travel_distance_weekend', 'travel_aim',
        'cost', 'cost_alacarte', 'car_km_month',
        'factor1', 'factor2', 'factor3', 'factor4', 'factor6',
        'taz', 'district',
        'MaasFamiliar', 'first_car', 'first_taxi', 'first_pt',
        'distance5', 'normal_depart',
        'Al_taxi', 'Al_PT', 'Al_bike', 'Al_sharedbike', 'Carown',
        'c6', 'c7',
        'trips_per_month',
    ]

    agents = {}
    for col in agent_cols:
        if col in df.columns:
            agents[col] = df[col].values.astype(np.float32)
        else:
            print(f"  Warning: column '{col}' not found, using zeros")
            agents[col] = np.zeros(len(df), dtype=np.float32)

    # TAZ as int for bincount
    agents['taz'] = df['taz'].values.astype(np.int32)
    # Ensure non-negative TAZ
    agents['taz'] = np.clip(agents['taz'], 0, None)

    # Weight: each agent represents ~25 people
    agents['weight'] = np.full(len(df), AGENT_WEIGHT, dtype=np.float32)

    N = len(df)
    print(f"  Created agent population: {N} agents, {len(agents)} variables")
    mem_mb = sum(v.nbytes for v in agents.values()) / 1e6
    print(f"  Memory usage: {mem_mb:.1f} MB")

    return agents


if __name__ == '__main__':
    agents = create_agent_population()
    print(f"\nAgent population summary:")
    for k, v in agents.items():
        if k in ['sex', 'age1', 'income1', 'have_car', 'week_metro', 'factor1']:
            print(f"  {k}: mean={v.mean():.3f}, std={v.std():.3f}, "
                  f"min={v.min():.3f}, max={v.max():.3f}")
