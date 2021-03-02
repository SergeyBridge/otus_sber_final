import numpy as np
from catboost.utils import get_gpu_device_count

params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'thread_count': -1,
    'custom_metric': ['AUC:hints=skip_train~false', 'F1'],
    'task_type': 'CPU' if get_gpu_device_count() > 0 else 'CPU',
    # 'task_type': 'GPU', # if torch.cuda.is_available() else 'CPU',
    'grow_policy': 'Lossguide',   # 'SymmetricTree',  #  'Depthwise',
    # 'auto_class_weights': 'Balanced',
    'langevin': True,  # CPU only
    'iterations': 20,
    'learning_rate': 0.002,   # 4e-3,
    'l2_leaf_reg': 1e-1,
    'depth': 16,
    'max_leaves': 10,
    'border_count': 128,
    'verbose': 1,
    'od_type': 'Iter',
    'od_wait': 100,
    # 'early_stopping_rounds': 100,

    # random control
    'bootstrap_type': 'Bayesian',
    # 'random_seed': 100,
    'random_strength': 0.001,
    'rsm': 1,
    'bagging_temperature': 0,
    'boosting_type': 'Plain',   # 'Ordered'
}


dtypes = {
    'DealDate': np.datetime64,
    'ValueDate': np.datetime64,
    'MaturityDate': np.datetime64,
    'Deal_characteristics_1': np.float32,
    'Deal_characteristics_2': np.float32,
    'Deal_characteristics_3': np.float32,
    'Deal_characteristics_4': np.int32,
    'Client_characteristics_1': np.int32,
    'Client_characteristics_2': np.int32,
    'target': np.int32,
}

categorical_dtypes_ft = {
    'Deal_characteristics_4': np.int32,
    'Client_characteristics_1': np.int32,
}

numeric_dtypes_ft = {
    'Deal_characteristics_1': np.float32,
    'Deal_characteristics_2': np.float32,
    'Deal_characteristics_3': np.float32,
    'Client_characteristics_2': np.int32,

}

dates_dtypes_ft = {
    'DealDate': np.datetime64,
    # 'ValueDate': np.datetime64,
    'MaturityDate': np.datetime64,
}

dates = ['DealDate', 'ValueDate', 'MaturityDate']
