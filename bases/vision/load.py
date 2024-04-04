from os.path import join
import torch
import torchvision
import torchvision.transforms as transforms
from bases.vision.data_loader import DataLoader
from bases.vision.transforms import Flatten, OneHot, DataToTensor
from configs import femnist, celeba, cifar10, imagenet100
from bases.vision.datasets import FEMNIST, CelebA, TinyImageNet, ImageNet100
from func import *
__all__ = ["get_data", "get_data_loader"]


def get_config_by_name(name: str):
    if name.lower() == "femnist":
        return femnist
    elif name.lower() == "celeba":
        return celeba
    elif name.lower() == "cifar10":
        return cifar10
    elif name.lower() in ["imagenet100", "imagenet-100", "imagenet_100"]:
        return imagenet100
    else:
        raise ValueError("{} is not supported.".format(name))


def get_data(name: str, data_type, transform=None, target_transform=None, user_list=None):
    dataset = get_config_by_name(name)

    if dataset == femnist:
        assert data_type in ["train", "test"]
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        if target_transform is None:
            target_transform = transforms.Compose(
                [DataToTensor(dtype=torch.long), OneHot(dataset.NUM_CLASSES, to_float=True)])

        return FEMNIST(root=join("datasets", "FEMNIST"), train=data_type == "train", download=True, transform=transform,
                       target_transform=target_transform, user_list=user_list)

    elif dataset == celeba:
        assert data_type in ["train", "test"]
        if transform is None:
            transform = transforms.Compose([transforms.Resize((84, 84)),
                                            transforms.ToTensor()])
        if target_transform is None:
            target_transform = transforms.Compose([DataToTensor(dtype=torch.long)])

        return CelebA(root=join("datasets", "CelebA"), train=data_type == "train", download=True, transform=transform,
                      target_transform=target_transform, user_list=user_list)

    elif dataset == cifar10:
        assert data_type in ["train", "test"]
        if transform is None:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            if data_type == "train":
                transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.RandomCrop(32, 4),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
            else:
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
        if target_transform is None:
            target_transform = transforms.Compose([DataToTensor(dtype=torch.long),
                                                   OneHot(dataset.NUM_CLASSES, to_float=True)])

        return torchvision.datasets.CIFAR10(root=join("datasets", "CIFAR10"), train=data_type == "train", download=True,
                                            transform=transform, target_transform=target_transform)

    elif dataset == imagenet100:
        assert data_type in ["train", "val"]
        if transform is None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            if data_type == "train":
                transform = transforms.Compose([transforms.RandomResizedCrop(imagenet100.IMAGE_SIZE),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
            else:
                transform = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(imagenet100.IMAGE_SIZE),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])

        return ImageNet100(root=join("datasets", "ImageNet100"), train=data_type == "train", transform=transform,
                           target_transform=None)

    else:
        raise ValueError("{} dataset is not supported.".format(name))


# def get_data_loader(name: str, data_type, batch_size=None, shuffle: bool = False, sampler=None, transform=None,
#                     target_transform=None, subset_indices=None, num_workers=8, pin_memory=True, user_list=None):
#     assert data_type in ["train", "val", "test"]
#     if data_type == "train":
#         assert batch_size is not None, "Batch size for training data is required"
#     if shuffle is True:
#         assert sampler is None, "Cannot shuffle when using sampler"

#     data = get_data(name, data_type=data_type, transform=transform, target_transform=target_transform,
#                     user_list=user_list)

#     if subset_indices is not None:
#         data = torch.utils.data.Subset(data, subset_indices)
#     if data_type != "train" and batch_size is None:
#         batch_size = len(data)

#     return DataLoader(data, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
#                       pin_memory=pin_memory)


# # DATA LOADER FOR SINGLE MODALITY
# class data_loader(object):
#     def __init__(self, train_val_test,X_lidar_train,X_lidar_test,y_train,y_test):
#         if train_val_test == 'train':
#             self.feat = X_lidar_train
#             self.label = y_train
#         elif train_val_test == 'val':
#             self.feat = X_lidar_validation
#             self.label = y_validation
#         elif train_val_test == 'test':
#             self.feat = X_lidar_test
#             self.label = y_test
#         print(train_val_test)

#     def __len__(self):
#         return self.feat.shape[0]

#     def __getitem__(self, index):
#         feat = self.feat[index] #
#         label = self.label[index] # change
#         return torch.from_numpy(feat).type(torch.FloatTensor), torch.from_numpy(label).type(torch.FloatTensor)

# DATALOADER FOR THREE MODALITY FUSION
class data_loader(object):
    def __init__(self, ds1, ds2, ds3, label):
        self.ds1 = ds1
        self.ds2 = ds2
        self.ds3 = ds3
        self.label = label

    def __getitem__(self, index):
        x1, x2, x3 = self.ds1[index], self.ds2[index],  self.ds3[index]
        label = self.label[index]
        return torch.from_numpy(x1).type(torch.FloatTensor), torch.from_numpy(x2).type(torch.FloatTensor),  torch.from_numpy(x3).type(torch.FloatTensor), torch.from_numpy(label).type(torch.FloatTensor)

    def __len__(self):
        return self.ds1.shape[0]  # assume both datasets have same length



def get_data_loader(name: str, X_lidar_train,X_lidar_test,X_img_train,X_img_test,X_coord_train,X_coord_test,y_train,y_test,client,data_type, batch_size=None, shuffle: bool = True, sampler=None, transform=None,
                    target_transform=None, subset_indices=None, num_workers=8, pin_memory=True, user_list=None):

    # print('**********genrating train/validation data************')
    # clients = ['0','1','2','3','4','5','6','7','8','9']
    # selected_paths = detecting_related_file_paths("/home/batool/FL/data_half_half_size/",['Cat1','Cat2','Cat3','Cat4'],clients)
    # ########################RF
    # RF_train, RF_val = get_data_files(selected_paths,'rf','rf')
    # y_train,num_classes = custom_label(RF_train,'one_hot')
    # y_validation, _ = custom_label(RF_val,'one_hot')
    # ####Lidar
    # X_lidar_train, X_lidar_validation = get_data_files(selected_paths,'lidar','lidar')

    # print('****************Loading test set*****************')
    # RF_test = open_npz('/home/batool/FL/baseline_code/all_test/'+'/'+'rf'+'_'+'all.npz','rf')
    # X_lidar_test =  open_npz('/home/batool/FL/baseline_code/all_test/'+'/'+'lidar'+'_'+'all.npz','lidar')
    # y_test, _ = custom_label(RF_test,'one_hot')
    # print('test set shape',X_lidar_test.shape)
    # print('******************Succesfully generated the data*************************')
    lens_of_each_client = [2910, 3054, 2904, 2887, 2965, 2873, 2706, 2881]#, 2732, 2724]# added
    if data_type == 'train':
        # begin = 0   # for centerlizec
        # end = sum(lens_of_each_client)
        begin = sum(lens_of_each_client[:client])
        end = sum(lens_of_each_client[:(client+1)])
        print('CHECK POINT',client,begin,end,len(X_lidar_train[begin:end]),len(X_img_train[begin:end]),len(X_coord_train[begin:end]),len(y_train[begin:end]))
        data = data_loader(X_lidar_train[begin:end],X_img_train[begin:end],X_coord_train[begin:end],y_train[begin:end])  # for test is not important
    elif data_type == 'test':
        data = data_loader(X_lidar_test,X_img_test,X_coord_test,y_test)  # for test is not important

    print('sampler',type(sampler))
    return DataLoader(data, batch_size=32, shuffle=True, sampler=sampler, num_workers=num_workers,
                      pin_memory=pin_memory)



