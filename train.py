import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import warnings
warnings.filterwarnings("ignore")
import torch
num_workers = 8
torch.set_num_threads(num_workers)
import logging
from glob import glob
import pandas as pd
import random
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from model.H2ASeg import H2ASeg
from utils.metrics import show_deep_metrics, metrics_tensor
from monai.data import list_data_collate, DataLoader
from datetime import datetime
import monai
from monai.transforms import (
    CropForegroundd,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    ToTensord,
    AddChanneld,
    RandRotated,
)
import time



def main():
    patch_size = (128, 128, 64)
    batch_size = 2 # Note: Each case takes two images —— num_images = batch_size * 2
    num_channels = [8, 16, 32, 64, 128]
    window_size = [8, 8, 4]
    pretrained_model_path = ""
    use_pretrained_model = False
    loss_weights = [5, 4, 3, 2, 1]

    # set file path
    date = datetime.now().strftime("%m_%d")
    ct_path = "/data3/clh/datasets/hecktor2022_spac_norm/imagesTr"
    pet_path = "/data3/clh/datasets/hecktor2022_spac_norm/imagesTr"
    label_path = "/data3/clh/datasets/hecktor2022_spac_norm/labelsTr"
    images_CT = sorted(glob(os.path.join(ct_path, "*CT.nii.gz")))
    images_PET = sorted(glob(os.path.join(pet_path, "*PT.nii.gz")))
    labels = sorted(glob(os.path.join(label_path, "*.nii.gz")))
    
    # set save path
    save_path = f"./save/hecktor2022_H2ASeg/{date}/"
    os.makedirs(save_path, exist_ok=True)

    # dataset split
    split = 0.6
    split_1 = 0.8
    
    # setting of model's input
    in_channels = 1
    n_classes = 2
    
    # setting of epoch
    epochs = 200
    save_model_interval = 10
    val_interval = 10 
    
    # adjustment of learning rate
    lr = 1e-4
    step_size = 1
    gamma = 0.99
    
    # fixing random seed
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    
    # print metric
    deep = True
    
    # dice & bceloss
    criterion_dice = monai.losses.DiceLoss(include_background=True, 
                                           to_onehot_y=True, 
                                           softmax=True).cuda()
    criterion_ce = nn.CrossEntropyLoss().cuda()
    
    # data transform
    train_transforms = Compose(
            [
                LoadImaged(keys=["img", "img_ct", "seg"]),
                AddChanneld(keys=["img", "img_ct","seg"]),
                CropForegroundd(keys=["img","img_ct","seg"], source_key="img", select_fn=lambda x:x>x.min()),
                RandCropByPosNegLabeld(
                        keys=["img","img_ct","seg"],
                        label_key="seg",
                        spatial_size=patch_size,
                        pos=1,
                        neg=1,
                        num_samples=2,
                    ),
                RandRotated(keys=['img', 'img_ct', 'seg'], prob=0.8, range_z=45),
                ToTensord(keys=["img", "img_ct"]),
            ]
        )

    val_transforms = Compose(
            [
                LoadImaged(keys=["img", "img_ct", "seg"]),
                AddChanneld(keys=["img", "img_ct", "seg"]),
                CropForegroundd(keys=["img","img_ct","seg"], source_key="img", select_fn=lambda x:x>x.min()),

                RandCropByPosNegLabeld(
                    keys=["img","img_ct","seg"],
                    label_key="seg",
                    spatial_size=patch_size,
                    pos=1,
                    neg=1,
                    num_samples=2,
                ),
                ToTensord(keys=["img", "img_ct"]),
            ]
        )
              
    length = len(images_CT)
    print("The number of samples:", length)

    # split dataset
    train_images = images_PET[: int(split * length)]
    train_ct_images = images_CT[: int(split * length)]
    train_labels = labels[: int(split * length)]

    test_images = images_PET[int(split * length):int(split_1*length)]
    test_ct_images = images_CT[int(split * length):int(split_1*length)]
    test_labels = labels[int(split * length):int(split_1*length)]

    train_files = [{"img": img, "img_ct": ct,"seg": seg} \
                    for img, ct, seg in zip(train_images, train_ct_images, train_labels)]
    val_files = [{"img": img, "img_ct": ct,"seg": seg} \
                    for img, ct, seg in zip(test_images, test_ct_images, test_labels)]
    print("Training set includes:", len(train_images))
    print("Validation set includes:", len(test_images))
    
    # create dataloader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
        )

    # create model and load pretrained weight
    model = H2ASeg(
                patch_size=patch_size,
                in_channels=in_channels,
                n_classes=n_classes,
                num_channels=num_channels,
                window_size=window_size).cuda()
    if use_pretrained_model:
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict(checkpoint)
        print(f'load checkpoint from {pretrained_model_path}\n')

    model.cuda()
    
    # Ready for training
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=lr / 10)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    epoch_loss_metric_list = []
    writer = SummaryWriter(save_path + 'logs')

    # training start
    print("*" * 20, "|| Training ||", "*" * 20)
    iteration = 0
    best_dice = 0
    best_train_dice = 0
    for epoch in range(epochs):
        start = time.time()
        model.train()
        loss_metric_list = []
        
        for step, batch_data in enumerate(train_loader):
            iteration += 1
            ct, pet, labels = batch_data["img_ct"].type(torch.FloatTensor).cuda(),\
                              batch_data["img"].type(torch.FloatTensor).cuda(),\
                              batch_data["seg"].type(torch.LongTensor).cuda()
            optimizer.zero_grad()
            outputs = model(ct, pet)
            labels[labels!=0] = 1
            
            # calculate loss
            loss1 = criterion_ce(outputs[0], labels.squeeze(1)) + criterion_dice(input=outputs[0], target=labels)
            loss2 = criterion_ce(outputs[1], labels.squeeze(1)) + criterion_dice(input=outputs[1], target=labels)
            loss3 = criterion_ce(outputs[2], labels.squeeze(1)) + criterion_dice(input=outputs[2], target=labels)
            loss4 = criterion_ce(outputs[3], labels.squeeze(1)) + criterion_dice(input=outputs[3], target=labels)
            loss5 = criterion_ce(outputs[4], labels.squeeze(1)) + criterion_dice(input=outputs[4], target=labels)
            loss = loss1 * loss_weights[0] + loss2 * loss_weights[1] + loss3 * loss_weights[2] + \
                        loss4 * loss_weights[3] + loss5 * loss_weights[4]
            loss.backward()
            optimizer.step()
            l = loss.item()
            l1 = loss1.item()
            l2 = loss2.item()
            l3 = loss3.item()
            l4 = loss4.item()
            l5 = loss5.item()
            
            epoch_len = len(train_ds) // batch_size
            # print metric
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {epoch+1}/{epochs} {step}/{epoch_len} ")
            print(f"[Training Loss:{l:.4f}, Loss1:{l1:.4f}, Loss2:{l2:.4f}, Loss3:{l3:.4f}, Loss4:{l4:.4f}, Loss5:{l5:.4f}]")
            
            fp, fn, iou, dice = show_deep_metrics(outputs, labels, deep)

            writer.add_scalar("Training Loss", l, iteration)
            writer.add_scalar("Training Loss1", l1, iteration)
            writer.add_scalar("Training Loss2", l2, iteration)
            writer.add_scalar("Training Loss3", l3, iteration)
            writer.add_scalar("Training Loss4", l4, iteration)
            writer.add_scalar("Training Loss5", l5, iteration)
            writer.add_scalar('Training FP', fp, iteration)
            writer.add_scalar('Training FN', fn, iteration)
            writer.add_scalar('Training Dice', dice, iteration)
            loss_metric_list += [[l, fp, fn, iou, dice]]
        scheduler.step()

        # save average value of [loss, fp, fn, iou, dice] in every epoch
        epoch_loss_metric_list += [np.mean(loss_metric_list, axis=0).tolist()]

        if (epoch+1) % save_model_interval == 0:
            torch.save(model.state_dict(), save_path + f"{epoch}.pth")
        
        now_dice = epoch_loss_metric_list[-1][-1]
        if now_dice >= best_train_dice:
            print(f"get new best dice {best_train_dice} -> {now_dice}, save new 'train_best.pth'")
            best_train_dice = now_dice 
            torch.save(model.state_dict(), save_path + "train_best.pth")
        
        print(f"training epoch {epoch + 1}: best training_epoch_dice == {best_train_dice}")
        print(f"training epoch {epoch + 1}: average [loss:{epoch_loss_metric_list[-1][0]}, ", end="")
        print(f"fp:{epoch_loss_metric_list[-1][1]}, fn:{epoch_loss_metric_list[-1][2]}, ", end="")
        print(f"iou:{epoch_loss_metric_list[-1][3]}, dice:{epoch_loss_metric_list[-1][4]}]")
        print(f"training epoch {epoch + 1}: time cost {time.time() - start} s")

        # validation
        if (epoch + 1) % val_interval == 0:
            print("\n"*2, "*" * 20, "|| Validating ||", "*" * 20)
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                metric_list = []
                for val_data in val_loader:
                    val_ct, val_pet, val_labels = val_data["img_ct"].type(torch.FloatTensor).cuda(),\
                                    val_data["img"].type(torch.FloatTensor).cuda(),\
                                    val_data["seg"].type(torch.LongTensor).cuda()
                    r = model(val_ct, val_pet)[0]
                    val_labels[val_labels!=0] = 1
                    # compute metric for current iteration
                    r = r.argmax(dim=1, keepdim=True)
                    fp, fn, iou, dice = metrics_tensor(val_labels, r)
                    print("val {} [FP:{:.4f}, FN:{:.4f}, IoU:{:.4f}, Dice:{:.4f} pix:{:6}/{:6}]"\
                            .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), fp, fn, iou, dice, r.sum(), val_labels.sum()))
                                                        
                    metric_list += [[fp, fn, iou, dice]]

            mean_metric = np.mean(metric_list, axis=0)

            now_dice = mean_metric[3]
            if now_dice >= best_dice:
                print(f"get new best dice {best_dice} -> {now_dice}, save new 'val_best.pth'")
                best_dice = now_dice
                torch.save(model.state_dict(), save_path + "val_best.pth")

            print(f"epoch {epoch + 1}/{epochs}")
            print(f"epoch {epoch + 1}: best validating_epoch_dice == {best_dice}")
            print("[FP:{:.4f}, FN:{:.4f}, IoU:{:.4f}, Dice:{:.4f}]".format(mean_metric[0], 
                                                                           mean_metric[1], 
                                                                           mean_metric[2], 
                                                                           mean_metric[3]))
            writer.add_scalar("Validating FP", mean_metric[0], epoch)
            writer.add_scalar('Validating FN', mean_metric[1], epoch)
            writer.add_scalar('Validating IoU', mean_metric[2], epoch)
            writer.add_scalar('Validating Dice', mean_metric[3], epoch)
            
            print("*" * 20, "|| Final Validating ||", "*" * 20, "\n"*2)
    writer.close()


if __name__ == "__main__":
    main()