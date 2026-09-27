import time
import os
import torch
import pandas as pd
from torch.optim import Adam
from scipy.stats import norm
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import numpy as np
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from model.ema import EMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset.data_load import train_loader, val_loader, test_loader

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")


from model.mix import GRMP

model = GRMP(hidden_channels=128, g_cutoff=7.0, cutoff=5.0, num_spherical=3, num_radial=6, envelope_exponent=5, 
             basis_emb_size_dist=128, basis_emb_size_angle=256, basis_emb_size_torsion=512, emb_channels=256, 
             pass_layers=3, adaptive_type='learned').to(device)

loss_func = torch.nn.L1Loss()

def test(model, test_loader, ema, device, use_ema=True):
    model.eval()
    
    if use_ema and ema is not None:
        ema.assign(model)
    
    total_loss = 0
    total_mae = 0
    total_samples = 0
    
    all_preds = []
    all_targets = []
    all_smiles = []  
    
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            out = model(batch_data)
            
            target = batch_data.M.unsqueeze(1)
            
            if hasattr(batch_data, 'smiles'):
                all_smiles.extend(batch_data.smiles)
            
            loss_batch = F.l1_loss(out, target, reduction='sum').item()
            mae_batch = (out - target).abs().sum().item()
            
            total_loss += loss_batch
            total_mae += mae_batch
            total_samples += target.size(0)
            
            all_preds.append(out.detach().cpu())
            all_targets.append(target.detach().cpu())
    
    if use_ema and ema is not None:
        ema.resume(model)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    
    return avg_loss, avg_mae, all_targets, all_preds, all_smiles  # 返回SMILES
    
def smp(device, train_loader, valid_loader, test_loader, model, loss_func, 
        epochs=500, batch_size=64, lr=0.0005, lr_decay_factor=0.5, 
        lr_decay_step_size=50, weight_decay=0,
        
        save_dir = '/export/home/zhengym/model/fusion/GRMP/out/abaltion/all/model_save',
        log_dir = '/export/home/zhengym/model/fusion/GRMP/out/abaltion/all/log',
        save_path = '/export/home/zhengym/model/fusion/GRMP/out/abaltion/all/picts'):
    
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)
    ema = EMA(model, decay=0.999)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)    
    os.makedirs(log_dir, exist_ok=True)
    # os.makedirs(csv_path, exist_ok=True)

    epoch_mae_log = os.path.join(log_dir, 'density.log')
    
    train_size = len(train_loader.dataset)
    
    best_val_mae = float('inf')
    best_val_loss = float('inf')      
    best_epoch = 0
    
    train_loss_list = []
    val_loss_list = []
    train_mae_list = []
    val_mae_list = []
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        step = 0
        model.train()
        total_train_loss = 0
        total_train_mae = 0
        total_samples = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            target = data.M.unsqueeze(1)

            batch_size_curr = target.size(0)
            loss = loss_func(output, target)  
            total_train_loss += loss.item() * batch_size_curr
            total_train_mae += (output - target).abs().sum().item()
            total_samples += batch_size_curr
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000, norm_type=2)
            optimizer.step()
            
            ema(model)
            step += 1
        scheduler.step()

        train_loss = total_train_loss / total_samples
        train_mae = total_train_mae / total_samples
        train_loss_list.append(train_loss)
        train_mae_list.append(train_mae)
        
        val_loss, val_mae, _, _, _ = test(model, valid_loader, ema, device, use_ema=True)
        val_loss_list.append(val_loss)
        val_mae_list.append(val_mae)
        
        epoch_time = time.time() - t0

        with open(epoch_mae_log, 'a') as f:
            f.write(f'Epoch: {epoch}, Train Loss:{train_loss:.4f}, Train MAE: {train_mae:.6f}, Val Loss:{val_loss:.4f}, Val MAE: {val_mae:.6f}, epoch_time: {epoch_time:.2f} s\n')
        
        # 保存最佳模型
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_loss = val_loss
            best_epoch = epoch
            
            # 应用EMA后保存
            ema.assign(model)
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_mae': val_mae
            }, os.path.join(save_dir, 'best_density.pt'))
            ema.resume(model)
    # 记录
    with open(epoch_mae_log, 'a') as f: 
        f.write(f'Best Epoch: {best_epoch}\tBest valid Loss: {best_val_loss:.6f}\tBest valid MAE: {best_val_mae:.6f}\n')

    # 加载最佳模型
    checkpoint = torch.load(os.path.join(save_dir, 'best_density.pt'))
    model.load_state_dict(checkpoint['model'])
    
    test_loss, test_mae, all_preds, all_targets, test_smiles = test( model, test_loader, ema=None, device=device, use_ema=False)
    
    test_mse = F.mse_loss(all_preds, all_targets).item()
    test_rmse = torch.sqrt(torch.tensor(test_mse)).item()
    
    preds_np = all_preds.cpu().numpy().flatten()
    targets_np = all_targets.cpu().numpy().flatten()
    test_r2 = r2_score(targets_np, preds_np)

    with open(epoch_mae_log, 'a') as f: 
        f.write(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.6f}, Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}\n')
    
    def plot_combined_metrics(train_losses, val_losses, targets, preds, epochs, save_path, filename):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.plot(range(1, epochs + 1), train_losses, label='Train Loss', linewidth=2)
        ax1.plot(range(1, epochs + 1), val_losses, label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')

        legend = ax1.legend()
        for text in legend.get_texts():
            text.set_fontweight('bold') 
  
        ax1.tick_params(axis='both', width=1.5, length=4, labelsize=12)  

        for label in ax1.get_xticklabels():
            label.set_fontweight('bold')  
        for label in ax1.get_yticklabels():
            label.set_fontweight('bold')  
          
        targets_np = targets.cpu().numpy().flatten()
        preds_np = preds.cpu().numpy().flatten()
        
        test_mae = mean_absolute_error(targets_np, preds_np)
        test_mse = mean_squared_error(targets_np, preds_np)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(targets_np, preds_np)
        
        errors = preds_np - targets_np
        absolute_errors = np.abs(errors)
        
        scatter = ax2.scatter(targets_np, preds_np,  alpha=0.6, s=30, c=absolute_errors, 
                                cmap='coolwarm', edgecolors='black', linewidth=0.5)
        
        ax2.set_aspect('auto')
        cbar = plt.colorbar(scatter, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('AE', fontsize=12, fontweight='bold', labelpad=10)
        cbar.ax.tick_params(labelsize=12)
        for label in cbar.ax.get_yticklabels():  
            label.set_fontweight('bold')

        min_val = min(targets_np.min(), preds_np.min())
        max_val = max(targets_np.max(), preds_np.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
        
        metrics_text = f'Test MAE = {test_mae:.4f} g/cm3\nRMSE = {test_rmse:.4f} g/cm3\nR² = {test_r2:.4f}'
        # metrics_text = f'Test MAE = {test_mae:.4f} kcal/mol\nRMSE = {test_rmse:.4f} kcal/mol\nR² = {test_r2:.4f}'               # 1

        # metrics_text = f'Test MAE = {test_mae:.4f} cal/g\nRMSE = {test_rmse:.4f} cal/g\nR² = {test_r2:.4f}'
        # metrics_text = f'Test MAE = {test_mae:.4f} km/s\nRMSE = {test_rmse:.4f} km/s\nR² = {test_r2:.4f}'
                
        # metrics_text = f'Test MAE = {test_mae:.4f} GPa\nRMSE = {test_rmse:.4f} GPa\nR² = {test_r2:.4f}'
        
        ax2.tick_params(axis='both', width=1.5, length=4, labelsize=12)   

        for label in ax2.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax2.get_yticklabels():
            label.set_fontweight('bold')

        ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes,  
                verticalalignment='top', fontsize=11, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
        ax2.set_xlabel('True Density (g/cm³)', fontsize=18, fontweight='bold')
        ax2.set_ylabel('Pred Density (g/cm³)', fontsize=18, fontweight='bold')

        # ax2.set_xlabel('True HOF (kcal/mol)', fontsize=18, fontweight='bold')
        # ax2.set_ylabel('Pred HOF (kcal/mol)', fontsize=18, fontweight='bold')

        # ax2.set_xlabel('True Detonation Q (cal/g)', fontsize=18, fontweight='bold')
        # ax2.set_ylabel('Pred Detonation Q (cal/g)', fontsize=18, fontweight='bold')

        # ax2.set_xlabel('True Detonation D (Km/s)', fontsize=18, fontweight='bold')
        # ax2.set_ylabel('Pred Detonation (Km/s)', fontsize=18, fontweight='bold')
        
        # ax2.set_xlabel('True Detonation P (GPa)', fontsize=18, fontweight='bold')
        # ax2.set_ylabel('Pred Detonation P (GPa)', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        
        save_file_path = os.path.join(save_path, filename)
        fig.savefig(save_file_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)

        # df_results = pd.DataFrame({
        #     'smiles': test_smiles,
        #     'true_value': targets_np,
        #     'predicted_value': preds_np,
        #     'absolute_error': absolute_errors,
        #     'relative_error': (absolute_errors / np.abs(targets_np)) * 100
        # })
        
        # # 按绝对误差降序排序
        # df_sorted = df_results.sort_values('absolute_error', ascending=False)
        
        # # 保存前50个异常点
        # path_out=os.path.join(csv_path,'density_outliers.csv')
        # df_sorted.head(50).to_csv(path_out, index=False)
        
        # threshold = test_mae * 3
        # df_outliers = df_results[df_results['absolute_error'] > threshold]

        # if len(df_outliers) > 0:
        #     out_path = os.path.join(csv_path, f'density_threshold_{threshold:.4f}.csv')
        #     df_outliers.to_csv(out_path, index=False)
        # else:
        #     print(f"没有误差超过 {threshold:.4f} 的异常点")
        
        # print("测试集误差最大的10个样本:")
        # print(df_sorted[['smiles','true_value', 'predicted_value', 'absolute_error', 'relative_error']].head(10).to_string())

    plot_combined_metrics(train_loss_list, val_loss_list, all_targets, all_preds, epochs, save_path, 'density.png')
    
    return best_val_mae, test_mae, test_rmse, test_r2

loss_func = torch.nn.L1Loss()

smp(device, train_loader, val_loader, test_loader, model, loss_func, 
    epochs=100, batch_size=64, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15)