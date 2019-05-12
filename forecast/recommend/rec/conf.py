# -*- coding: utf-8 -*-

feature_columns = ['sid', 'click_mode', 'o1', 'o2', 'd1', 'd2',
                   'first_mode', 'max_dist', 'min_dist', 'mean_dist', 'std_dist', 'max_price', 'min_price',
                   'mean_price', 'std_price', 'max_eta', 'min_eta', 'mean_eta', 'std_eta',
                   'max_dist_mode', 'min_dist_mode', 'max_price_mode', 'min_price_mode',
                   'max_eta_mode', 'min_eta_mode', 'mode_feas_1', 'mode_feas_2',
                   'mode_feas_3', 'mode_feas_4', 'mode_feas_5', 'mode_feas_6',
                   'mode_feas_7', 'mode_feas_8', 'mode_feas_9', 'mode_feas_10',
                   'mode_feas_11', 'svd_mode_0', 'svd_mode_1', 'svd_mode_2', 'svd_mode_3',
                   'svd_mode_4', 'svd_mode_5', 'svd_mode_6', 'svd_mode_7', 'svd_mode_8',
                   'svd_mode_9', 'svd_fea_0', 'svd_fea_1', 'svd_fea_2', 'svd_fea_3',
                   'svd_fea_4', 'svd_fea_5', 'svd_fea_6', 'svd_fea_7', 'svd_fea_8',
                   'svd_fea_9', 'svd_fea_10', 'svd_fea_11', 'svd_fea_12', 'svd_fea_13',
                   'svd_fea_14', 'svd_fea_15', 'svd_fea_16', 'svd_fea_17', 'svd_fea_18',
                   'svd_fea_19', 'weekday', 'hour', 'o20', 'd20', 'num_direct_distance','same_cls20']

cate_columns = ['max_dist_mode', 'min_dist_mode', 'max_price_mode',
                'min_price_mode', 'max_eta_mode', 'min_eta_mode', 'first_mode', 'weekday', 'hour', 'o20', 'd20','same_cls20']


#############
feature_columns_bkp = ['sid', 'click_mode', 'o1', 'o2', 'd1', 'd2',
                   'first_mode', 'max_dist', 'min_dist', 'mean_dist', 'std_dist', 'max_price', 'min_price',
                   'mean_price', 'std_price', 'max_eta', 'min_eta', 'mean_eta', 'std_eta',
                   'max_dist_mode', 'min_dist_mode', 'max_price_mode', 'min_price_mode',
                   'max_eta_mode', 'min_eta_mode', 'mode_feas_1', 'mode_feas_2',
                   'mode_feas_3', 'mode_feas_4', 'mode_feas_5', 'mode_feas_6',
                   'mode_feas_7', 'mode_feas_8', 'mode_feas_9', 'mode_feas_10',
                   'mode_feas_11', 'svd_mode_0', 'svd_mode_1', 'svd_mode_2', 'svd_mode_3',
                   'svd_mode_4', 'svd_mode_5', 'svd_mode_6', 'svd_mode_7', 'svd_mode_8',
                   'svd_mode_9', 'svd_fea_0', 'svd_fea_1', 'svd_fea_2', 'svd_fea_3',
                   'svd_fea_4', 'svd_fea_5', 'svd_fea_6', 'svd_fea_7', 'svd_fea_8',
                   'svd_fea_9', 'svd_fea_10', 'svd_fea_11', 'svd_fea_12', 'svd_fea_13',
                   'svd_fea_14', 'svd_fea_15', 'svd_fea_16', 'svd_fea_17', 'svd_fea_18',
                   'svd_fea_19', 'weekday', 'hour']

cate_columns_bkp = ['max_dist_mode', 'min_dist_mode', 'max_price_mode',
                'min_price_mode', 'max_eta_mode', 'min_eta_mode', 'first_mode', 'weekday', 'hour']
