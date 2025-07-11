#!/usr/bin/env python
# coding: utf-8

# In[2]:


#pip install pyscipopt


# In[3]:


pip install geopy


# In[2]:


from pyscipopt import Model, quicksum, multidict
import numpy as np
import pandas as pd
import geopandas as gpd
import random
import json
from geopy.distance import geodesic
import sys

sys.stdout = open("CFLP_test_output.log", "w")
sys.stderr = sys.stdout


# In[4]:


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
        model.addCons(quicksum(x[i,j] for i in I) >= 0.6 * M[j] * y[j], "MinCapacityUse(%s)"%j) # ensures no school has capacity under 60%
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


# In[8]:


# for I, d make a dictionary of planning units to number of students
pu = gpd.read_file('/Users/leahwallihan/Durham_school_planning/DPS-Planning/GIS_files/hs_full_geo.geojson').set_index('pu_2324_84')
pu = pu.to_crs('EPSG:4326')


# In[7]:


pu_data = pu['basez'].to_dict()
I, d = multidict(pu_data)


# In[10]:


# for model with half SGRs:
#pu_half_SGR = pu.copy()
#pu_half_SGR['basez+gen'] = pu['basez'] + 0.15*pu['student_gen']

#pu_data = pu_half_SGR['basez+gen'].to_dict()
#I, d = multidict(pu_data)


# In[12]:


# for J, M make a dictionary of sites to capacities
schools = gpd.read_file('/Users/leahwallihan/Durham_school_planning/DPS-Planning/GIS_files/dps_hs_locations.geojson')
schools = schools.to_crs('EPSG:4326')

# find which planning units have existing school
schools['pu'] = None

for i, geometry in enumerate(pu['geometry']):
    in_geometry = geometry.contains(schools['geometry'])
    pu_id = pu.index[i]

    schools.loc[in_geometry, 'pu'] = pu_id


# In[14]:


# let's remove planning units in downtown from J to make problem simpler
not_central = pu[(pu['Region'] != 'Central')]

# initialize dictionary of planning units with capacity of 1600 for potential site
pu_dict = {}
for idx, row in not_central.iterrows():
    pu_dict[idx] = 1550

# replace capacities of planning units with existing schools
pu_dict[45] = 1300
pu_dict[507] = 1510
pu_dict[602] = 1240 # reduce by 300 for choice?
pu_dict[566] = 1240
pu_dict[290] = 1235 # reduce by 300 for choice?

J, M = multidict(pu_dict)

# define which sites already exist
existing_sites = {602, 290, 45, 566, 507}


# In[16]:


# Get centroids and convert to lat/lon tuples
centroid_coords = {
    idx: (geom.y, geom.x)  # (latitude, longitude)
    for idx, geom in pu.geometry.centroid.items()
}

# Now build the distance matrix using geodesic distances
c = {}
for i in I:
    for j in J:
        c[i, j] = geodesic(centroid_coords[i], centroid_coords[j]).miles


# In[11]:



# for testing:
I_small = random.sample(I, 100)
d_small = {i: d[i] for i in I_small}
c_small = {(i,j): c[i,j] for i in I_small for j in J if (i,j) in c}

model = flp(I_small, J, d_small, M, c_small, existing_sites=existing_sites)
x,y = model.data
model.setParam('limits/solutions', 3)
model.optimize()
EPS = 1.e-6
edges = [(i,j) for (i,j) in x if model.getVal(x[i,j]) > EPS]
facilities = [j for j in y if model.getVal(y[j]) > EPS]
print ("Optimal value=", model.getObjVal())
print ("Facilities at nodes:", facilities)
print ("Edges:", edges)




# In[ ]:


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


# In[118]:


for report in solution_reports:
    print(f"\n--- Solution #{report['solution_number']} ---")
    print("Facilities opened:", report['facilities'])

    print("Assignments:")
    for fac, pus in report['assignments'].items():
        print(f"  Facility {fac} <-- Planning Units {pus}")

    print("Student Count per Facility:")
    student_counts = {}
    for fac in report['facilities']:
        count = 0
        for pu_id in report['assignments'][fac]:
            count += pu.loc[pu_id, 'basez']
        student_counts[fac] = count

    report['student_count'] = student_counts

    for fac, count in student_counts.items():
        print(f"  Facility {fac}: {count} students")



# In[88]:


pu_new = pu.copy()

for solution in solution_reports: 
    facility_to_pus = solution['assignments']

    pu_to_facility = {
        pu_id: facility
        for facility, pu_list in facility_to_pus.items()
        for pu_id in pu_list
    }

    pu_new['assignment'] = pu.index.map(pu_to_facility)
    solution_number = solution['solution_number']
    pu_new.to_file(f"CFLP_test{solution_number}.geojson", driver="GeoJSON")


# In[94]:


with open('CFLP_test.json', 'w') as f:
    json.dump(solution_reports, f)


# In[ ]:







