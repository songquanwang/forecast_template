from geopy.distance import geodesic
import pandas as pd


def gen_od_feas(data):
    data['num_o1'] = data['o'].apply(lambda x: float(x.split(',')[0]))
    data['num_o2'] = data['o'].apply(lambda x: float(x.split(',')[1]))
    data['num_d1'] = data['d'].apply(lambda x: float(x.split(',')[0]))
    data['num_d2'] = data['d'].apply(lambda x: float(x.split(',')[1]))

    locations = pd.read_csv('../../input/kdd2019_regular/phase1/lntlat_adress_6525.csv')
    location_columns = ['adcode', 'district', 'lntlat', 'street', 'street_number']
    locations = locations[location_columns]

    data['cat_source_lntlat'] = data.o
    data['cat_des_lntlat'] = data.d

    locations.columns = map(lambda x: 'cat_source_' + x, location_columns)
    merge = pd.merge(data, locations, on=['cat_source_lntlat'], how='inner')

    locations.columns = map(lambda x: 'cat_des_' + x, location_columns)

    merge = pd.merge(merge, locations, on=['cat_des_lntlat'], how='inner')

    merge['cat_same_district'] = 1
    merge['cat_same_street_number'] = 1
    merge['cat_same_adcode'] = 1

    merge.loc[merge['cat_source_district'] != merge['cat_des_district'], 'cat_same_district'] = 0
    merge.loc[merge['cat_source_street_number'] != merge['cat_des_street_number'], 'cat_same_street_number'] = 0
    merge.loc[merge['cat_source_adcode'] != merge['cat_des_adcode'], 'cat_same_adcode'] = 0
    merge['num_direct_distance'] = merge.apply(lambda x: geodesic((x.num_o2, x.num_o1), (x.num_d2, x.num_d2)).m, axis=1)

    #
    from sklearn.preprocessing import LabelEncoder
    merge[['cat_source_district', 'cat_des_district']] = merge[['cat_source_district', 'cat_des_district']].apply(LabelEncoder().fit_transform)

    merge = merge.drop(['o', 'd', 'cat_source_lntlat', 'cat_des_lntlat', 'cat_source_street', 'cat_source_street_number', 'cat_des_street', 'cat_des_street_number'], axis=1)
    return merge


merge_df=pd.read_csv('./forecast/recommend/data/features/gbdt/features_od.csv')
o='39.96,116.32'
d='39.79,116.33'


o='116.32,39.96'
d='116.33,39.79'
def get_dis(o, d):
    o1, o2 = float(o.split(',')[1]), float(o.split(',')[0])
    d1, d2 = float(d.split(',')[1]), float(d.split(',')[0])
    return geodesic((o1, o2), (d1, d2)).m
merge_df['num_direct_distance'] = merge_df.apply(lambda x: get_dis(x['o'],x['d']), axis=1)

newport_ri = (41.49008, -71.312796)
cleveland_oh = (41.499498, -81.695391)
print(geodesic(newport_ri, cleveland_oh).miles)


newport_ri = (39.96, 116.32)
cleveland_oh = (39.79, 116.33)
print(geodesic(newport_ri, cleveland_oh).miles)


round(merge_df,7).to_csv('./forecast/recommend/data/features/gbdt/features_od.csv',index=False)