from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpContinuous, PULP_CBC_CMD
import numpy as np
import pandas as pd
import geopandas as gpd
import json
from geopy.distance import geodesic
import sys

sys.stdout = open("CFLP_fullSGR_pulp.log", "w")
sys.stderr = sys.stdout

# Load data
pu = gpd.read_file('/hpc/group/dataplus/lnw20/CFLP_infiles/hs_full_geo.geojson').set_index('pu_2324_84')
pu = pu.to_crs('EPSG:4326')

# Adjust demand for half SGR
pu['basez+gen'] = pu['basez'] + 0.3 * pu['student_gen']
d = pu['basez+gen'].to_dict()
I = list(d.keys())

# Load schools and determine existing sites
schools = gpd.read_file('/hpc/group/dataplus/lnw20/CFLP_infiles/dps_hs_locations.geojson').to_crs('EPSG:4326')
schools['pu'] = None
for i, geometry in enumerate(pu['geometry']):
    in_geometry = geometry.contains(schools['geometry'])
    pu_id = pu.index[i]
    schools.loc[in_geometry, 'pu'] = pu_id

not_central = pu[pu['Region'] != 'Central']
J = list(not_central.index)
M = {idx: 1550 for idx in J}
M.update({45: 1400, 507: 1510, 602: 1340, 566: 1240, 290: 1335})
existing_sites = {602, 290, 45, 566, 507}

# Build distance matrix
centroids = pu.geometry.centroid
coords = {idx: (geom.y, geom.x) for idx, geom in centroids.items()}
c = {(i, j): geodesic(coords[i], coords[j]).miles for i in I for j in J}

# Model
model = LpProblem("CFLP", LpMinimize)
x = {(i, j): LpVariable(f"x_{i}_{j}", lowBound=0, cat=LpContinuous) for i in I for j in J}
y = {j: LpVariable(f"y_{j}", cat=LpBinary) for j in J}
BIG_M = sum(d.values())

# Objective
model += lpSum(c[i, j] * x[i, j] for i in I for j in J)

# Constraints
for i in I:
    model += lpSum(x[i, j] for j in J) == d[i]

for j in J:
    model += lpSum(x[i, j] for i in I) <= 1.05 * M[j] * y[j]
    model += lpSum(x[i, j] for i in I) >= 0.7 * M[j] * y[j]

for i in I:
    for j in J:
        model += x[i, j] <= BIG_M * y[j]

for j in existing_sites:
    model += y[j] == 1

model += lpSum(y[j] for j in J) <= 6

# Solve
solver = PULP_CBC_CMD(msg=1)
model.solve(solver)

# Collect solution
assignments = {j: [] for j in J if y[j].varValue > 0.5}
for (i, j), var in x.items():
    if var.varValue > 1e-3:
        if y[j].varValue < 0.5:
            print(f"WARNING: PU {i} assigned to CLOSED facility {j}")
        assignments[j].append(i)

# Student counts
student_counts = {
    j: sum(pu.loc[i, 'basez'] for i in pus)
    for j, pus in assignments.items()
}

# Output
solution = {
    "objective_value": model.objective.value(),
    "facilities": list(assignments.keys()),
    "assignments": assignments,
    "student_count": student_counts
}

with open("CFLP_fullSGR_pulp.json", "w") as f:
    json.dump(solution, f, indent=2)

# Export assignment map
pu_new = pu.copy()
pu_to_fac = {i: j for j, pus in assignments.items() for i in pus}
pu_new["assignment"] = pu.index.map(pu_to_fac)
pu_new.to_file("CFLP_fullSGR_pulp.geojson", driver="GeoJSON")
