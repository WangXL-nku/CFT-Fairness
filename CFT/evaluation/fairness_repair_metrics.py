import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.fairness_metrics import compute_deo, compute_aaod


def evaluate_model(model, dataloader, sensitive_attr, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_sens = []
    with torch.no_grad():
        for imgs, labels, sens in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = (outputs > 0).float() if outputs.size(1) == 1 else outputs.argmax(dim=1).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_sens.append(sens.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_sens = np.concatenate(all_sens)
    deo = compute_deo(all_labels, all_preds, all_sens)
    aaod = compute_aaod(all_labels, all_preds, all_sens)
    return deo, aaod


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, labels, _ in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        if outputs.size(1) == 1:
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
        else:
            loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_model", type=str, required=True)
    parser.add_argument("--train_orig", type=str, required=True)
    parser.add_argument("--adv_samples", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--sensitive_file", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_model", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 加载原始模型
    model = torch.load(args.orig_model, map_location=device)
    model.to(device)

    # 加载原始训练集和对抗样本，构建增强训练集
    train_orig = np.load(args.train_orig)  # 假设包含 imgs, labels
    adv = np.load(args.adv_samples)        # 假设包含 adv_imgs, labels
    sens = np.load(args.sensitive_file)    # 训练集敏感属性

    imgs_all = np.concatenate([train_orig['imgs'], adv['adv_imgs']], axis=0)
    labels_all = np.concatenate([train_orig['labels'], adv['labels']], axis=0)
    sens_all = np.concatenate([sens, sens[:len(adv['adv_imgs'])]], axis=0)  # 简单复用，实际需对应

    train_dataset = TensorDataset(
        torch.from_numpy(imgs_all).float(),
        torch.from_numpy(labels_all).float(),
        torch.from_numpy(sens_all).float()
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 测试集
    test_data = np.load(args.test_data)
    test_sens = np.load(args.sensitive_file.replace('train', 'test'))  # 假设
    test_dataset = TensorDataset(
        torch.from_numpy(test_data['imgs']).float(),
        torch.from_numpy(test_data['labels']).float(),
        torch.from_numpy(test_sens).float()
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 原始模型在测试集上的性能
    deo_orig, aaod_orig = evaluate_model(model, test_loader, test_sens, device)
    print(f"Original - DEO: {deo_orig:.4f}, AAOD: {aaod_orig:.4f}")

    # 重训练
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss() if train_dataset.tensors[1].dim() == 1 else nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    deo_retrain, aaod_retrain = evaluate_model(model, test_loader, test_sens, device)
    print(f"Retrained - DEO: {deo_retrain:.4f}, AAOD: {aaod_retrain:.4f}")
    print(f"ΔDEO: {deo_orig - deo_retrain:.4f}, ΔAAOD: {aaod_orig - aaod_retrain:.4f}")

    if args.output_model:
        torch.save(model, args.output_model)