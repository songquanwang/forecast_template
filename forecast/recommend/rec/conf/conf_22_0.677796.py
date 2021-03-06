# od + svd profile

"""
min基本全0 可以去掉
"""
od_features = [
    # 'sid',  'd', 'o', 'pid','plan_time', 'click_mode',
    #  'is_rain' 'is_rain_max_mode',
    'o1', 'o2', 'd1', 'd2',
    # 'o10', 'd10', 'same_cls10',
    'o20', 'd20', 'same_cls20',
    # 'same_cls20',
    # 'o30', 'd30', 'same_cls30'
    'num_direct_distance',
    # 'dis_cat_y'

    # 'o10_max_mode', 'd10_max_mode',
    'o20_max_mode', 'd20_max_mode'
    # 'o30_max_mode', 'd30_max_mode'
]
#  'hour',
time_features = ['weekday', 'hour', 'is_week_end', 'month', 'dayofyear'  #

                 ]
profile_svd_features = [
    # profile svd
    'svd_fea_0', 'svd_fea_1', 'svd_fea_2', 'svd_fea_3', 'svd_fea_4', 'svd_fea_5',
    'svd_fea_6', 'svd_fea_7', 'svd_fea_8', 'svd_fea_9', 'svd_fea_10', 'svd_fea_11',
    'svd_fea_12', 'svd_fea_13', 'svd_fea_14', 'svd_fea_15',
    'svd_fea_16', 'svd_fea_17', 'svd_fea_18', 'svd_fea_19'
    # 'p_cls'
]

# 没有 'pid'
# 'pid_min_dist','pid_min_price','pid_min_dj',  'pid_max_sd_dj', 'pid_min_sd_dj', 'pid_mean_sd_dj', 'pid_std_sd_dj',    'pid_max_mode',  # cat
pid_ext_features = [
    'pid',
    'last_mode', 'pre_mode',
    'pid_max_mode',
    'pid_max_dist', 'pid_mean_dist',
    'pid_max_price', 'pid_mean_price',
    'pid_max_eta', 'pid_min_eta', 'pid_mean_eta',
    'pid_max_dj', 'pid_mean_dj',
    'pid_max_sd', 'pid_min_sd',
    'pid_std_sd', 'pid_std_dist', 'pid_std_price', 'pid_std_eta', 'pid_std_dj', 'pid_mean_sd',
    # # 各个模式占比
    'mode_num_0', 'mode_num_1', 'mode_num_2', 'mode_num_3', 'mode_num_4',
    'mode_num_5', 'mode_num_6', 'mode_num_7', 'mode_num_8', 'mode_num_9',
    'mode_num_10', 'mode_num_11'
]

plans_features = [
    'mode_feas_0', 'mode_feas_1', 'mode_feas_2', 'mode_feas_3',
    'mode_feas_4', 'mode_feas_5', 'mode_feas_6', 'mode_feas_7',
    'mode_feas_8', 'mode_feas_9', 'mode_feas_10', 'mode_feas_11',
    'first_mode', 'second_mode', 'last_mode_1',
    'max_dist', 'min_dist', 'mean_dist', 'std_dist',
    'max_price', 'min_price', 'mean_price', 'std_price',
    'max_eta', 'min_eta', 'mean_eta', 'std_eta',
    'max_dj', 'min_dj', 'mean_dj', 'std_dj',
    'max_sd', 'min_sd', 'mean_sd', 'std_sd',
    'max_sd_dj', 'min_sd_dj', 'mean_sd_dj', 'std_sd_dj',
    'max_price_eta', 'min_price_eta', 'mean_price_eta', 'std_price_eta',
    'max_dist_mode', 'min_dist_mode',
    'max_price_mode', 'min_price_mode',
    'max_eta_mode', 'min_eta_mode',
    'max_sd_mode', 'min_sd_mode',
    'max_dj_mode', 'min_dj_mode',
    'max_sd_dj_mode', 'min_sd_dj_mode',
    'max_price_eta_mode', 'min_price_eta_mode',
    'svd_mode_0', 'svd_mode_1', 'svd_mode_2', 'svd_mode_3',
    'svd_mode_4', 'svd_mode_5', 'svd_mode_6', 'svd_mode_7', 'svd_mode_8',
    'svd_mode_9'

]

feature_columns = od_features + time_features + profile_svd_features + pid_ext_features + plans_features

cate_columns = ['max_dist_mode', 'min_dist_mode',
                'max_price_mode', 'min_price_mode',
                'max_eta_mode', 'min_eta_mode',
                'max_sd_mode', 'min_sd_mode',
                'max_dj_mode', 'min_dj_mode',
                'max_sd_dj_mode', 'min_sd_dj_mode',
                'max_price_eta_mode', 'min_price_eta_mode',

                'last_mode', 'pre_mode',
                'first_mode', 'second_mode', 'last_mode_1',
                'weekday', 'hour', 'is_week_end',
                'o20', 'd20', 'o20_max_mode', 'd20_max_mode', 'same_cls20', 'pid'

                ]
