#import packages
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.stdout = open("gravity_model.log", "w")
sys.stderr = sys.stdout

hs_full_geo = gpd.read_file('/hpc/group/dataplus/lnw20/CFLP_infiles/hs_full_geo.geojson').to_crs(epsg=3857)

dps_base = gpd.read_file('/hpc/group/dataplus/lnw20/CFLP_infiles/dps_base_2324.geojson')
dps_base = dps_base.to_crs(epsg = 3857)
base_hs = {'Jordan High School','Riverside High School','Northern High School','Hillside High School','Southern High School'}
dps_base_hs = dps_base[dps_base['name'].isin(base_hs)]
dps_base_hs = dps_base_hs[['name','geometry']]
dps_base_hs.loc[5,'name']='New High School'
dps_base_hs = dps_base_hs.reset_index()

dps_pu = gpd.read_file('/hpc/group/dataplus/lnw20/CFLP_infiles/pu_2324_SPLIT.geojson').rename(columns={'pu_2324_848':'pu_2324_84'})
dps_pu = dps_pu.to_crs(epsg = 3857).sort_values(by='pu_2324_84')

school_names = ['Southern High School','Hillside High School','Northern High School','Riverside High School','Jordan High School','New High School']
capacities = [1340,1335,1400,1240,1510,1550]

local_hs_full_geo = hs_full_geo.copy()
local_dps_base_hs = dps_base_hs.copy()

def score_candidate(candidate,sgr,lower_bound,upper_bound):
    i=0
    counts = pd.DataFrame({'school':school_names,
                           'capacity':capacities,
                           'count':[0,0,0,0,0,0],
                           'adjust':[0,0,0,0,0,0]
                            })                              

    
    candidate_geom = candidate['geometry']
    local_dps_base_hs.loc[5,'geometry']=candidate_geom.centroid
    counts['pct_capacity'] = counts['count']/counts['capacity']
    
    
    while ((counts['pct_capacity']<lower_bound/100)|(counts['pct_capacity']>upper_bound/100)).any() and i<=149:                                    
        assignments = []
        for pu in local_hs_full_geo.itertuples(index=False):                   
            centroid = pu.geometry.centroid
            pu_scores = []
            for j in range(6):    
                dist = centroid.distance(local_dps_base_hs.loc[j,'geometry'])
                score = dist + counts.loc[j,'adjust']
                pu_scores.append(score)
            assign = school_names[pu_scores.index(min(pu_scores))]
            assignments.append(assign)
        local_hs_full_geo['assign'] = assignments
    
        for j,school in enumerate(school_names):
            assigned_students = int(local_hs_full_geo.loc[local_hs_full_geo['assign'] == school, 'basez'].sum() + local_hs_full_geo.loc[local_hs_full_geo['assign'] == school, 'student_gen'].sum()*sgr/100)
            counts.loc[j, 'count'] = assigned_students  
                
            if counts.loc[j,'count'] <= counts.loc[j,'capacity']*lower_bound/100:
                counts.loc[j,'adjust'] -= 200
            elif counts.loc[j,'count'] >= counts.loc[j,'capacity']*upper_bound/100:
                counts.loc[j,'adjust'] += 200
            else:
                pass
        counts['pct_capacity'] = counts['count']/counts['capacity']
        i+=1

    if i<=150:
        objective=local_hs_full_geo.merge(local_dps_base_hs,left_on='assign',right_on='name',how='left')
        objective['distance']=objective['geometry_x'].distance(objective['geometry_y'])
        objective_score = ((objective['basez']+sgr*objective['student_gen']/100)*objective['distance']).sum()/(10**7)

    else:
        objective_score = 10

    return objective_score

dps_pu['score'] = dps_pu.apply(lambda row:score_candidate(row,12,70,100),axis=1)

dps_pu.to_file('gravity_model_sol.geojson', driver='GeoJSON')