import os
from config import get_config
import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import utils
import wandb
import models
import data_utils
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from torch import nn
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score, mean_absolute_percentage_error
import matplotlib as mpl

# Default settings
mpl.rcParams.update(mpl.rcParamsDefault)
torch.set_float32_matmul_precision('high')
plt.style.use("ggplot")

def train_model(model, optimizer, scheduler, num_epochs=50, seq_to_seq=False, model_name="best_model"):
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            model.train(True)
            train_loss = 0
            for X_batch, y_batch in tqdm(train):
                optimizer.zero_grad()
                # print(X_batch[:, :, 5:].shape)
                y_pred = model(X_batch.to(device))
                if not seq_to_seq:
                    loss = loss_fn(y_pred.squeeze(), y_batch.to(device))
                else:
                    target = X_batch[:, :, -1:]
                    target = target[:, 1:, :]
                    loss = loss_fn(y_pred[:, :-1, :], target.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train)
            train_losses.append(train_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                for X_batch, y_batch in valid:
                    y_pred = model(X_batch.to(device))
                    if not seq_to_seq:
                        loss = loss_fn(y_pred.squeeze(), y_batch.to(device))
                    else:
                        target = X_batch[:, :, -1:]
                        target = target[:, 1:, :]
                        loss = loss_fn(y_pred[:, :-1, :], target.to(device))
                    val_loss += loss.item()

                val_loss /= len(valid)
                print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

            wandb.log({
                "Epoch" : epoch + 1,
                "Train Loss" : train_loss,
                "Validation Loss" : val_loss,
                'learning_rate' : optimizer.param_groups[0]['lr']
            })
            scheduler.step(val_loss)
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{model_name}.pth")
                wandb.save(f"{model_name}.pth")
        model.load_state_dict(
            torch.load(
                f"{model_name}.pth",
                weights_only=True
            )
        )
        return model, optimizer, scheduler, train_losses, val_losses

if __name__ == "__main__":
    config = get_config()
    quantiles_features = [f'q_{percent}' for percent in range(10, 100, 10)]
    utils.set_seed(42)
    
    name = (
        config['model_type'] +
        config['coefs_path']
        .replace('data/', '_')
        .replace('coefs_4comp_300window', '4cmp_300ws_')
        .replace('coefs_3comp_300window', '3cmp_300ws_')
        .replace('full.parquet', f"{config['data_params']['seq_len']}lags")
    )
    
    if config['use_ab'] == 'quantile':
        name = 'ab_' + name
    if config['use_fourier']:
        name = 'diff_fourier_' + name
    if config['process_coefs'] == 'cdf':
        name = 'cdf_' + name 
    if config['process_coefs'] == 'quantile':
        name = 'quantile_' + name
        
    wandb.login()
    wandb.init(
        name=name,
        project="Thesis data 95-25",
        config = config
    )

    wandb.save("*.py")
    print(name)
    
    #==================PREPROCESS=====================

    data = pd.read_parquet(config['dataset_path'])
    coefs = pd.read_parquet(config['coefs_path'])
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].astype('float32')

    numeric_cols = coefs.select_dtypes(include=['number']).columns
    coefs[numeric_cols] = coefs[numeric_cols].astype('float32')

    data_diff = pd.DataFrame()
    data_diff[config['column']] = data[config['column']].dropna().diff().dropna()
    data_diff = data_diff.join(data[['is_nan']])
    if config['use_fourier']:
        data_diff = data_diff.join(data[['fourier_predictor']].diff().dropna())
    
    new_coefs = None
    if config['process_coefs'] == 'cdf':
        new_coefs = utils.ProcessCoefsCDF(coefs.values, n_points=5)
        new_coefs = pd.DataFrame(new_coefs, columns=[f'{i}_cdf' for i in range(1, 5 + 1)])
    if config['process_coefs'] == 'quantile':
        new_coefs = utils.ProcessCoefs(coefs.values)
        new_coefs = pd.DataFrame(new_coefs, columns=quantiles_features)

    if config['use_ab']:
        ab_coefs = utils.ProcessCoefsAB(coefs.values)
        ab_coefs = pd.DataFrame(ab_coefs, columns=['a_bias', 'b_diffusion'])
        new_coefs = (
            new_coefs.join(ab_coefs)
            if new_coefs is not None
            else ab_coefs
        )
    df = None
    if new_coefs is not None:
        df = (
            new_coefs
            .merge(
                data_diff.dropna().reset_index(drop=True),
                left_index=True,
                right_index=True)
        )
    else:
        df = data_diff.dropna().reset_index(drop=True)
    
    if config['model_type'].startswith('EncodeEM') or config['use_matrix_features']:
        df = coefs.merge(df, left_index=True, right_index=True)
    # print(df)
    # print(*df.columns)
    # df = df.query('is_nan == 0')
    #==================TRAIN MODEL=====================
    
    device = 'cuda'
    df = df.iloc[config['data_params']['subsample_shift'] : int(df.shape[0] * config['data_params']['subsample_rate']), :]
    TRAIN_SIZE = int(df.shape[0] * config['data_params']['train_ratio'])
    VALID_SIZE = int(df.shape[0] * config['data_params']['valid_ratio'])
    TEST_SIZE = int(df.shape[0] * config['data_params']['test_ratio'])
    SEQ_LENGTH = config['data_params']['seq_len']

    train_dataset = data_utils.DataTs(df.iloc[:TRAIN_SIZE], SEQ_LENGTH, config['column'], 1, config['use_matrix_features'])
    val_dataset = data_utils.DataTs(df.iloc[TRAIN_SIZE: TRAIN_SIZE + VALID_SIZE], SEQ_LENGTH, config['column'], 1, config['use_matrix_features'])
    test_dataset = data_utils.DataTs(df.iloc[TRAIN_SIZE + VALID_SIZE:], SEQ_LENGTH, config['column'], 1, config['use_matrix_features'])
    if 'input_size' in config['model_hyperparams']:
        config['model_hyperparams']['input_size'] = val_dataset[0][0].shape[1]
        wandb.config['model_hyperparams']['input_size'] = val_dataset[0][0].shape[1]

    train = DataLoader(train_dataset, batch_size=config['data_params']['batch_size'], shuffle=True)
    valid = DataLoader(val_dataset, batch_size=config['data_params']['batch_size'], shuffle=False)
    test = DataLoader(test_dataset, batch_size=config['data_params']['batch_size'], shuffle=False)
    
    model_type = getattr(models, config['model_type'])
    
    model = model_type(**config['model_hyperparams']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train_params']['lr'])

    loss_fn = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    model, optimizer, scheduler, train_losses, val_losses = (
        train_model(
            model,
            optimizer,
            scheduler,
            config['train_params']['epochs'],
            config['train_params']['seq_to_seq'],
            name
        )                                                  
    )
    
    #==================EVALUATE MODEL=====================
    
    model.eval()

    predictions = []
    true = []

    all_data = DataLoader(data_utils.DataTs(df, config['data_params']['seq_len'], config['column'], 1, config['use_matrix_features']), batch_size=config['data_params']['batch_size'])

    with torch.no_grad():
        for X_batch, y_batch in tqdm(all_data, desc="Inference"):
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            if config['train_params']['seq_to_seq']:
                y_pred = y_pred[:, -1, :]
            predictions.extend(y_pred.cpu().numpy())
            true.extend(y_batch.numpy())

    predictions = np.array(predictions)
    true = np.array(true)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    ax.plot(true, color='y', label='true')
    ax.plot(predictions, alpha=1, color='b', label='predict')
    ax.axvline(TRAIN_SIZE, -10, 10, color='r')
    ax.axvline(TRAIN_SIZE + VALID_SIZE, -10, 10, color='r')

    plt.legend(loc='lower left')
    wandb.log({"model prediction diff": wandb.Image(plt)})
    # ax.legend(title='models', loc='upper left', labels=['true', 'predict'])

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    integrated_true = true.cumsum()
    # print(f'{integrated_true.shape=}')
    # print(f'{predictions.shape=}')
    integrated_predictions = np.roll(integrated_true, 1)[1:] + predictions.ravel()[:-1]
    # print(f'{integrated_predictions.shape=}')
    ax.plot(integrated_true, color='y', label='true')
    ax.plot(integrated_predictions, alpha=0.5, color='b', label='predict')
    ax.axvline(TRAIN_SIZE, -10, 10, color='r')
    ax.axvline(TRAIN_SIZE + VALID_SIZE, -10, 10, color='r')
    plt.legend(loc='lower left')
    wandb.log({"model prediction": wandb.Image(plt)})
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.plot(integrated_true[TRAIN_SIZE + VALID_SIZE : TRAIN_SIZE + VALID_SIZE + 3000], color='y', label='true')
    ax.plot(integrated_predictions[TRAIN_SIZE + VALID_SIZE : TRAIN_SIZE + VALID_SIZE + 3000], alpha=0.5, color='b', label='predict')
    plt.legend(loc='lower left')
    wandb.log({"model prediction on TEST": wandb.Image(plt)})
    #==================METRICS=====================
    mse = mean_squared_error(true[TRAIN_SIZE + VALID_SIZE : ], predictions[TRAIN_SIZE + VALID_SIZE : ])
    print(f"Среднеквадратичная ошибка (MSE) на тесте данных: {mse}")

    r2 = r2_score(true[TRAIN_SIZE + VALID_SIZE : ], predictions[TRAIN_SIZE + VALID_SIZE : ])
    print(f"r2: {r2}")

    to_directions = np.vectorize(lambda x: int(x > 0))

    auc = roc_auc_score(to_directions(true[TRAIN_SIZE + VALID_SIZE : ]),
                    to_directions(predictions[TRAIN_SIZE + VALID_SIZE : ]))

    accuracy = accuracy_score(to_directions(true[TRAIN_SIZE + VALID_SIZE : ]),
                    to_directions(predictions[TRAIN_SIZE + VALID_SIZE : ]))
    print(f"auc: {auc}")
    
    mape = mean_absolute_percentage_error(true[TRAIN_SIZE + VALID_SIZE : ], predictions[TRAIN_SIZE + VALID_SIZE : ])

    wandb.run.summary.update({
        "best_loss_val" : min(val_losses),
        'mse_test' : mse,
        'r2_test' : r2,
        'auc_test' : auc,
        'accuracy' : accuracy,
        'mape' : mape
    })

    wandb.log({
        "best_val_loss": min(val_losses),
        "test_mse": mse,
        "test_r2": r2,
        "test_auc": auc,
        'accuracy' : accuracy,
        'mape' : mape
    })
    predictions.dump('predictions/' + name)
    metrics_table = wandb.Table(columns=["Metric", "Value"])
    metrics_table.add_data("Best Validation Loss", min(val_losses))
    metrics_table.add_data("MSE Test", mse)
    metrics_table.add_data("R2 Test", r2)
    metrics_table.add_data("ROC AUC Test classification", auc)
    metrics_table.add_data("MAPE", mape)
    metrics_table.add_data('ACCURACY classification', accuracy)
    wandb.log({"Metrics Table": metrics_table})
    wandb.finish()
