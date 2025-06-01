import os

def get_config():
    config = {
        'dataset_path' : os.path.join('data', 'bx_fourier_clean.parquet'),
        'coefs_path' : os.path.join('data', 'bx_coefs_4comp_300window_weights_linear_full.parquet'),
        'use_fourier': False,
        'use_ab' : False,
        'process_coefs' : None,
        'use_matrix_features': True,
        'model_type' : 'BaseLSTM',
        'column' : 'Bx',
        'train_params' : {
            'lr' : 3e-4,
            'epochs' : 25,
            'type' : 'adam',
            'scheduler' : {
                'type' : 'ReduceLROnPlateau',
                'patience' : 3,
                'factor' : 0.5
            },
            'seq_to_seq' : False
        },
        'model_hyperparams' : {
            'input_size' : 1  + 2 + 9 + 1 + 1,
            # 'seq_size' : 30,

#             'hidden_size': 32,
#             'num_layers' : 2,
            # 'embed_size' : 8,
            # 'n_components' : 4,

            'hidden_size1' : 32,
            'hidden_size2' : 16,
            'output_size' : 1,
            # 'conv_hidden' : 16
        },
        'data_params' : {
            'subsample_shift' : 750_000,
            'subsample_rate' : 1,
            'batch_size' : 32 * 4,
            'train_ratio' : 0.5,
            'valid_ratio' : 0.2,
            'test_ratio' : 0.3,
            'seq_len' : 30
        }
    }
    return config