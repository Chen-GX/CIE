import os, sys
import os.path as osp
import logging
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from tensorboardX import SummaryWriter
from loss_func import CudaCKA
from log import config_logging


def loss_func(args, o_logs, c_logs, co_logs, label, xo, xc):
    o_loss = F.nll_loss(o_logs, label)  # o是因果的部分
    co_loss = F.nll_loss(co_logs, label)
    cka = CudaCKA(device=args.device)
    if args.kernel == 'rbf':
        if args.idp_type == 'xo':
            idp_loss = cka.kernel_CKA(xo, xc, sigma=None)  # 计算o和c特征的独立性程度
        elif args.idp_type == 'o_logs':
            idp_loss = cka.kernel_CKA(o_logs, c_logs, sigma=None)
    elif args.kernel == 'linear':
        idp_loss = cka.linear_CKA(o_logs, c_logs)  # 计算o和c特征的独立性程度
    loss = args.o * o_loss + args.idp * idp_loss + args.co * co_loss
    if torch.isnan(loss):
        assert False
    return loss, o_loss, idp_loss, co_loss


def eval_step(args, model, eval_loader, device, is_validation=False):
    model.eval()
    total_rows, total_correct_o, total_correct_c, total_correct_co = 0, 0, 0, 0
    total_loss, total_loss_o, total_loss_idp, total_loss_co = 0, 0, 0, 0
    for batch in eval_loader:
        batch = batch.to(device)
        mask = batch.val_mask if is_validation else batch.test_mask
        with torch.no_grad():
            if is_validation:  # 验证阶段
                o_logs, c_logs, co_logs, xo, xc = model(batch.x, batch.edge_index)
            else:
                o_logs, c_logs, co_logs, xo, xc = model(batch.x, batch.test_edge_index)
            o_logs, c_logs, co_logs, xo, xc = o_logs[mask], c_logs[mask], co_logs[mask], xo[mask], xc[mask]
            label = batch.y[mask].view(-1)
            loss, o_loss, idp_loss, co_loss = loss_func(args, o_logs, c_logs, co_logs, label, xo, xc)

            # update values for reporting
            # loss
            total_loss += loss.item()
            total_loss_o += o_loss.item()
            total_loss_idp += idp_loss.item()
            total_loss_co += co_loss.item()

            # 准确率
            total_correct_o += o_logs.argmax(dim=1).eq(label).sum().item()
            total_correct_c += c_logs.argmax(dim=1).eq(label).sum().item()
            total_correct_co += co_logs.argmax(dim=1).eq(label).sum().item()
            total_rows += torch.sum(mask).item()

    return (
        total_loss / total_rows,
        total_loss_o / total_rows,
        total_loss_idp / total_rows,
        total_loss_co / total_rows,
        total_correct_o / total_rows,
        total_correct_c / total_rows,
        total_correct_co / total_rows,
    )


def getDataLoader(dataset, args):
    if args.dataset == 'PPI':
        assert False
    else:
        train_loader = DataLoader(dataset, args.batch_size, shuffle=False)
        val_loader = test_loader = train_loader
    return train_loader, val_loader, test_loader


def train_baseline(model_func, dataset, args):
    if sys.platform.lower() == 'win32':
        root = "\\".join(os.path.abspath("__file__").split('\\')[:-1])
    elif sys.platform.lower() == "linux":
        root = "/".join(os.path.abspath("__file__").split('/')[:-1])
    else:
        assert False

    root = osp.join(root, 'info')  # log文件全部存到info中
    # 文件信息记录格式
    log_save_path = os.path.join(root, "log")
    os.makedirs(log_save_path, exist_ok=True)
    config_logging(file_name=osp.join(log_save_path, f'{args.timestamp}.log'))

    # model save
    model_save_path = osp.join(root, f"models/{args.timestamp}")
    os.makedirs(model_save_path, exist_ok=True)

    logging.info(f"\n***** {args.model} *****")
    logging.info(args)
    logging.info("Best Model Path: %s", model_save_path)
    logging.info("******************************\n")

    # TensorBoard: Logging
    tb_logdir = os.path.join(root, f"tblog/{args.timestamp}")
    os.makedirs(tb_logdir, exist_ok=True)
    if args.tensorboard:
        writer = SummaryWriter(tb_logdir)

    model = model_func(dataset.num_features, dataset.num_classes).to(args.device)
    # 注意，这里的batch_size是图的个数，而不是节点的个数
    train_loader, val_loader, test_loader = getDataLoader([dataset], args)

    epoch_num = 0
    best_val_loss = np.inf
    best_val_acc = 0
    best_val_epoch = 0
    device = args.device

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # loss_func = nn.CrossEntropyLoss().to(device)
    # loss_func = nn.NLLLoss().to(device)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    if args.use_scheduler:
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        scheduler_num = 0

    for epoch in range(args.epochs):
        # train model
        model.train()
        total_rows, total_correct_o, total_correct_c, total_correct_co = 0, 0, 0, 0
        total_loss, total_loss_o, total_loss_idp, total_loss_co = 0, 0, 0, 0

        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()

            o_logs, c_logs, co_logs, xo, xc = model(batch.x, batch.edge_index)
            o_logs, c_logs, co_logs, xo, xc = (
                o_logs[batch.train_mask],
                c_logs[batch.train_mask],
                co_logs[batch.train_mask],
                xo[batch.train_mask],
                xc[batch.train_mask],
            )

            label = batch.y[batch.train_mask].view(-1)
            loss, o_loss, idp_loss, co_loss = loss_func(args, o_logs, c_logs, co_logs, label, xo, xc)

            loss.backward()
            optimizer.step()

            # update values for reporting
            # loss
            total_loss += loss.item()
            total_loss_o += o_loss.item()
            total_loss_idp += idp_loss.item()
            total_loss_co += co_loss.item()

            # 准确率
            total_correct_o += o_logs.argmax(dim=1).eq(label).sum().item()
            total_correct_c += c_logs.argmax(dim=1).eq(label).sum().item()
            total_correct_co += co_logs.argmax(dim=1).eq(label).sum().item()
            total_rows += torch.sum(batch.train_mask).item()

            if (step + 1) % args.train_print_steps == 0:
                logging.info(
                    "Train Epoch [{}] | Step [{} / {}], Loss:[{:.4f}={}*{:.4f}+{}*{:.4f}+{}*{:.4f}]".format(
                        epoch,
                        step,
                        len(train_loader),
                        loss.item(),
                        args.o,
                        o_loss.item(),
                        args.idp,
                        idp_loss.item(),
                        args.co,
                        co_loss.item(),
                    )
                )

        train_loss, train_loss_o, train_loss_idp, train_loss_co = (
            total_loss / total_rows,
            total_loss_o / total_rows,
            total_loss_idp / total_rows,
            total_loss_co / total_rows,
        )
        train_o_acc, train_c_acc, train_co_acc = total_correct_o / total_rows, total_correct_c / total_rows, total_correct_co / total_rows

        # Valid model
        val_loss, val_loss_o, val_loss_idp, val_loss_co, val_o_acc, val_c_acc, val_co_acc = eval_step(
            args, model, val_loader, device, is_validation=True
        )
        logging.info(f"Evaluating at epoch {epoch}")
        # scheduler.step()
        # TensorBoard
        if args.tensorboard:
            writer.add_scalars(
                "Loss/train_loss",
                {'train_loss': train_loss, 'train_loss_o': train_loss_o, 'train_loss_idp': train_loss_idp, 'train_loss_co': train_loss_co},
                epoch + 1,
            )
            writer.add_scalars(
                "Acc/train_acc", {'train_o_acc': train_o_acc, 'train_c_acc': train_c_acc, 'train_co_acc': train_co_acc}, epoch + 1
            )
            writer.add_scalars(
                "Loss/val_loss",
                {'val_loss': val_loss, 'val_loss_o': val_loss_o, 'val_loss_idp': val_loss_idp, 'val_loss_co': val_loss_co},
                epoch + 1,
            )
            writer.add_scalars("Acc/val_acc", {'val_o_acc': val_o_acc, 'val_c_acc': val_c_acc, 'val_co_acc': val_co_acc}, epoch + 1)

        # log epoch的信息
        logging.info(
            "Epoch [{}] | Train Loss:[{:.8f}={}*{:.8f}+{}*{:.8f}+{}*{:.8f}] | Train o_Acc: {:.4f}, c_Acc: {:.4f}, co_Acc: {:.4f} | Valid Loss:[{:.8f}={}*{:.8f}+{}*{:.8f}+{}*{:.8f}] | Valid o_Acc: {:.4f}, c_Acc: {:.4f}, co_Acc: {:.4f}".format(
                epoch,
                train_loss,
                args.o,
                train_loss_o,
                args.idp,
                train_loss_idp,
                args.co,
                train_loss_co,
                train_o_acc,
                train_c_acc,
                train_co_acc,
                val_loss,
                args.o,
                val_loss_o,
                args.idp,
                val_loss_idp,
                args.co,
                val_loss_co,
                val_o_acc,
                val_c_acc,
                val_co_acc,
            )
        )
        # if args.bias_type=='node':
        if args.crite == 'loss':
            if val_loss < best_val_loss:
                # if val_o_acc > best_val_acc:
                logging.info(
                    "New Best Eval Loss {:.8f} | o_Acc: {:.4f}, c_Acc: {:.4f}, co_Acc: {:.4f}".format(
                        val_loss, val_o_acc, val_c_acc, val_co_acc
                    )
                )

                if args.use_scheduler and epoch > args.scheduler_begin:
                    scheduler_num = 0

                epoch_num = 0
                best_val_loss = val_loss
                # best_val_acc = val_o_acc
                best_val_epoch = epoch
                # Save Model
                model.to(torch.device("cpu"))
                torch.save(model, osp.join(model_save_path, f'{args.model}.pt'))
                model.to(device)
            else:
                epoch_num += 1
                if args.use_scheduler and epoch > args.scheduler_begin:
                    scheduler_num += 1
        elif args.crite == 'acc':
            # if val_loss < best_val_loss:
            if val_o_acc > best_val_acc:
                logging.info(
                    "New Best Eval Loss {:.8f} | o_Acc: {:.4f}, c_Acc: {:.4f}, co_Acc: {:.4f}".format(
                        val_loss, val_o_acc, val_c_acc, val_co_acc
                    )
                )

                if args.use_scheduler and epoch > args.scheduler_begin:
                    scheduler_num = 0

                epoch_num = 0
                # best_val_loss = val_loss
                best_val_acc = val_o_acc
                best_val_epoch = epoch
                # Save Model
                model.to(torch.device("cpu"))
                torch.save(model, osp.join(model_save_path, f'{args.model}.pt'))
                model.to(device)
            else:
                epoch_num += 1
                if args.use_scheduler and epoch > args.scheduler_begin:
                    scheduler_num += 1
        # elif args.bias_type == 'struc':
        #     # if val_loss < best_val_loss:
        #     if val_o_acc > best_val_acc:
        #         logging.info(
        #             "New Best Eval Loss {:.8f} | o_Acc: {:.4f}, c_Acc: {:.4f}, co_Acc: {:.4f}".format(
        #                 val_loss, val_o_acc, val_c_acc, val_co_acc
        #             )
        #         )

        #         if args.use_scheduler and epoch > args.scheduler_begin:
        #             scheduler_num = 0

        #         epoch_num = 0
        #         # best_val_loss = val_loss
        #         best_val_acc = val_o_acc
        #         best_val_epoch = epoch
        #         # Save Model
        #         model.to(torch.device("cpu"))
        #         torch.save(model, osp.join(model_save_path, f'{args.model}.pt'))
        #         model.to(device)
        #     else:
        #         epoch_num += 1
        #         if args.use_scheduler and epoch > args.scheduler_begin:
        #             scheduler_num += 1

        # 调整学习率
        if args.use_scheduler:
            if args.scheduler_type == 1:
                scheduler.step()
                logging.info(f"当前学习率: {optimizer.state_dict()['param_groups'][0]['lr']}")
            elif args.scheduler_type == 2 and epoch > args.scheduler_begin:
                if scheduler_num > args.scheduler_threh:
                    scheduler.step()
                    scheduler_num = 0
                    # 打印当前学习率
                    logging.info(f"更新当前学习率: {optimizer.state_dict()['param_groups'][0]['lr']}")

        logging.info("Epoch {} completed!\n".format(epoch))

        if epoch_num > args.early_stopping:
            logging.info(f"Early stopping at {epoch}")
            break

    logging.info("***************** Training completed *****************")
    logging.info(f"Best Valid Epoch at {best_val_epoch}\n")
    # Test Model
    best_model = torch.load(osp.join(model_save_path, f'{args.model}.pt')).to(device)
    test_loss, test_loss_o, test_loss_idq, test_loss_co, test_o_acc, test_c_acc, test_co_acc = eval_step(
        args, best_model, test_loader, device, is_validation=False
    )
    logging.info("**************** Test *****************")
    logging.info(f'Load Best model from {model_save_path}')
    logging.info(f'TimeStamp: {args.timestamp}')
    logging.info(
        "Test Loss:[{:.8f}={}*{:.8f}+{}*{:.8f}+{}*{:.8f}] | Test o_Acc: {:.4f}, c_Acc: {:.4f}, co_Acc: {:.4f}".format(
            test_loss, args.o, test_loss_o, args.idp, test_loss_idq, args.co, test_loss_co, test_o_acc, test_c_acc, test_co_acc
        )
    )
    if args.save_result:
        with open(f'./{args.save_file_name}/{args.dataset}_{args.model}_{args.bias_type}_{args.level}.result', 'a') as f:
            # o_Acc c_Acc co_Acc
            f.write('{:.4f}\t{:.4f}\t{:.4f}\n'.format(test_o_acc, test_c_acc, test_co_acc))
    if args.tensorboard:
        writer.close()
