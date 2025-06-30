# %%
# -*- coding: utf-8 -*-
pip install pyscipopt
# %%

from pyscipopt import Model, quicksum, multidict
import numpy as np
import pandas as pd
import geopandas as gpd
import random
# %%

# adapted from https://scipbook.readthedocs.io/en/latest/flp.html
def flp(I,J,d,M,c,existing_sites=None):
    model = Model("flp")
    x,y = {},{}
    for j in J:
        y[j] = model.addVar(vtype="B", name="y(%s)"%j)
        for i in I:
            x[i,j] = model.addVar(vtype="C", name="x(%s,%s)"%(i,j))
    for i in I:
        model.addCons(quicksum(x[i,j] for j in J) == d[i], "Demand(%s)"%i)
    for j in M:
        model.addCons(quicksum(x[i,j] for i in I) <= M[j]*y[j], "Capacity(%s)"%i)
    for (i,j) in x:
        model.addCons(x[i,j] <= d[i]*y[j], "Strong(%s,%s)"%(i,j))
    
    if existing_sites:
        for j in existing_sites:
            model.addCons(y[j] == 1, name=f"ForceOpen({j})")

    model.addCons(quicksum(y[j] for j in J) <= 6, "FacilityLimit") 
            
    model.setObjective(
        quicksum(c[i,j]*x[i,j] for i in I for j in J),
        "minimize")
    model.data = x,y
    return model
# %%

# for I, d make a dictionary of planning units to number of students
pu_data = gpd.read_file('/Users/leahwallihan/Durham_school_planning/DPS-Planning/GIS_files/pu_with_proj_SPLIT.geojson').set_index('pu_2324_84')
pu_data = pu_data['final_proj'].to_dict()

I, d = multidict(pu_data)
# %%

# for J, M make a dictionary of sites to capacities
schools = gpd.read_file('/Users/leahwallihan/Durham_school_planning/DPS-Planning/GIS_files/dps_hs_locations.geojson')
schools = schools.to_crs('EPSG:4326')
pu = gpd.read_file('/Users/leahwallihan/Durham_school_planning/geospatial files/pu_shape_new.geojson')
pu = pu.to_crs('EPSG:4326')

for i, geometry in enumerate(pu['geometry']):
    in_geometry = geometry.contains(schools['geometry'])
    pu_id = pu.loc[i, 'OBJECTID']

    schools.loc[in_geometry, 'pu'] = pu_id
# %%

# let's remove planning units in the North from J to make problem simpler
not_north = pu[(pu['Region'] != 'Central')]

# initialize dictionary of planning units with capacity of 1600 for potential site
pu_dict = {}
for _, row in not_north.iterrows():
    pu_dict[row['OBJECTID']] = 1600

# find which planning units have existing school
schools['pu'] = None

for i, geometry in enumerate(pu['geometry']):
    in_geometry = schools.within(geometry)
    pu_id = pu.loc[i, 'OBJECTID']
    schools.loc[in_geometry, 'pu'] = pu_id

# replace capacities of planning units with existing schools
pu_dict[45] = 1600
pu_dict[507] = 1810
pu_dict[602] = 1540
pu_dict[566] = 1540
pu_dict[290] = 1535

J, M = multidict(pu_dict)

# define which sites already exist
existing_sites = {602, 290, 45, 566, 507}
# %%

# create distance matrix
c = {}

pu_centroids = pu.set_index('OBJECTID').geometry.centroid 

for i in I:
    for j in J:
        dist = pu_centroids[i].distance(pu_centroids[j])
        c[i, j] = dist
# %%
        
model = flp(I, J, d, M, c, existing_sites=existing_sites)
model.setParam('limits/solutions', 7)
model.optimize()
EPS = 1.e-6
x,y = model.data
edges = [(i,j) for (i,j) in x if model.getVal(x[i,j]) > EPS]
facilities = [j for j in y if model.getVal(y[j]) > EPS]
print ("Optimal value=", model.getObjVal())
print ("Facilities at nodes:", facilities)
print ("Edges:", edges)
        
# %%

solution_reports = []

# Get all stored solutions
sols = model.getSols()

for sidx, sol in enumerate(sols):
    assignments = {}

    for (i_, j_) in x:
        if model.getSolVal(sol, x[i_, j_]) > 0.5:
            if j_ not in assignments:
                assignments[j_] = []
            assignments[j_].append(i_)

    student_count = {}
    if 'students' in globals():  
        for j_, pus in assignments.items():
            student_count[j_] = sum(students.get(i_, 0) for i_ in pus)

    solution_reports.append({
        'solution_number': sidx + 1,
        'facilities': list(assignments.keys()),
        'assignments': assignments,
        'student_count': student_count if 'students' in globals() else None
    })
# %%
    
for report in solution_reports:
    print(f"\n--- Solution #{report['solution_number']} ---")
    print("Facilities opened:", report['facilities'])

    print("Assignments:")
    for fac, pus in report['assignments'].items():
        print(f"  Facility {fac} <-- Planning Units {pus}")

    if report['student_count']:
        print("Student Count per Facility:")
        for fac, count in report['student_count'].items():
            print(f"  Facility {fac}: {count} students")

