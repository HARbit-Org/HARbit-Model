from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import yaml

with open(r"F:\UPC\Tesis\HARbit-Model\src\models\config\callbacks.yaml", 'r') as file:
    config = yaml.safe_load(file)['config']

_early_config = config['early_stopping']
_reduce_cofig = config['ReduceLROnPlateau']

callbacks = [
    EarlyStopping(
        monitor                 = _early_config['monitor'],
        patience                = _early_config['patience'],
        restore_best_weights    = _early_config['restore_best_weights'],
        verbose                 = _early_config['verbose']
    ),
    ReduceLROnPlateau(
        monitor     = _reduce_cofig['monitor'],
        factor      = _reduce_cofig['factor'],
        patience    = _reduce_cofig['patience'],
        min_lr      = float(_reduce_cofig['min_lr']),
        verbose     = _reduce_cofig['verbose']
    )
]