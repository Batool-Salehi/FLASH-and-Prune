import os
import torch
import sys
# python3 experiments/CIFAR10/adaptive.py -a -i -s 0 -e adaptive
sys.path.insert(1, '/home/batool/PruneFL')
from bases.fl.simulation.adaptive import AdaptiveServer, AdaptiveClient, AdaptiveFL, parse_args
from bases.optim.optimizer import SGD
from torch.optim import lr_scheduler
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from control.algorithm import ControlModule
from bases.vision.sampler import FLSampler
from bases.nn.models.vgg import VGG11
from configs.cifar10 import *
import configs.cifar10 as config
from func import *

from utils.save_load import mkdir_save




class CIFAR10AdaptiveServer(AdaptiveServer):
    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, X_lidar_train,X_lidar_test,X_img_train,X_img_test,X_coord_train,X_coord_test,y_train,y_test,'8',data_type="test", batch_size=3287, num_workers=8, pin_memory=True)#B

    def init_clients(self):
        rand_perm = torch.randperm(NUM_TRAIN_DATA).tolist()
        indices = []
        # len_slice = NUM_TRAIN_DATA // num_slices

        # for i in range(num_slices):
        #     indices.append(rand_perm[i * len_slice: (i + 1) * len_slice])
        lens_of_each_client = [2910, 3054, 2904, 2887, 2965, 2873, 2706, 2881]#, 2732, 2724]
        for i in range(len(lens_of_each_client)):
            begin = sum(lens_of_each_client[:i])
            end = sum(lens_of_each_client[:(i+1)])
            create_range = [i for i in range(begin,end)]
            indices.append([create_range[t] for t in torch.randperm(len(create_range))])

        models = [self.model for _ in range(NUM_CLIENTS)]
        self.indices = indices
        return models, indices

    def init_control(self):
        self.control = ControlModule(self.model, config=config)

    def init_ip_config(self):
        self.ip_train_loader = get_data_loader(EXP_NAME, X_lidar_train,X_lidar_test,X_img_train,X_img_test,X_coord_train,X_coord_test,y_train,y_test,0,data_type="train", batch_size=CLIENT_BATCH_SIZE,
                                               subset_indices=self.indices[0][:IP_DATA_BATCH * CLIENT_BATCH_SIZE],
                                               shuffle=True, num_workers=8, pin_memory=True)

        self.ip_test_loader = get_data_loader(EXP_NAME, X_lidar_train,X_lidar_test,X_img_train,X_img_test,X_coord_train,X_coord_test,y_train,y_test,'8',data_type="test", batch_size=3287, num_workers=8,
                                              pin_memory=True) #B
        # print('self.ip_test_loader',self.ip_test_loader)
        # ip_optimizer = torch.optim.Adam(self.model.parameters(), lr=INIT_LR, weight_decay=0.001)
        ip_optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        self.ip_optimizer_wrapper = OptimizerWrapper(self.model, ip_optimizer)
        self.ip_control = ControlModule(model=self.model, config=config)

    def save_exp_config(self):
        exp_config = {"exp_name": EXP_NAME, "seed": args.seed, "batch_size": CLIENT_BATCH_SIZE,
                      "num_local_updates": NUM_LOCAL_UPDATES, "mdd": MAX_DEC_DIFF, "init_lr": INIT_LR,
                      "lrhl": LR_HALF_LIFE, "ahl": ADJ_HALF_LIFE, "use_adaptive": self.use_adaptive,
                      "client_selection": args.client_selection}
        if self.client_selection:
            exp_config["num_users"] = num_users
        mkdir_save(exp_config, os.path.join(self.save_path, "exp_config.pt"))



class CIFAR10AdaptiveClient(AdaptiveClient):
    def init_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=INIT_LR, weight_decay=0.001)
        self.optimizer_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5 ** (1 / LR_HALF_LIFE))
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, self.optimizer_scheduler)

    def init_train_loader(self, tl):
        self.train_loader = tl

    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, X_lidar_train,X_lidar_test,X_img_train,X_img_test,X_coord_train,X_coord_test,y_train,y_test,'8',data_type="test", batch_size=3287, num_workers=8, pin_memory=True) #B



if __name__ == "__main__":

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    # torch.cuda.set_device(3)
    args = parse_args()
    torch.manual_seed(args.seed)

    num_users = 8 #10 #B
    num_slices = num_users if args.client_selection else NUM_CLIENTS


    print('**********genrating train data************')
    experiment_epiosdes = ['0','1','2','3','4','5','6','7','8','9']
    experiment_catergories = ['Cat1','Cat2','Cat3','Cat4']
    data_folder = "/home/batool/FL/data_half_half_size/"       ####path to dataset
    clients_data = {}
    lens_of_each_client = []
    for clients in tqdm(experiment_epiosdes):
        selected_paths = detecting_related_file_paths(data_folder,experiment_catergories,clients)
        ########################RF
        RF_train_client = get_data_files(selected_paths,'rf','rf')
        # y_train_client,num_classes = custom_label(RF_train_client,'one_hot')   # appret;y regrssion
        y_train_client = RF_train_client
        ################GPS
        X_coord_train_client = get_data_files(selected_paths,'gps','gps')
        X_coord_train_client = X_coord_train_client / 9747
        ## For convolutional input
        X_coord_train_client = X_coord_train_client.reshape((X_coord_train_client.shape[0], X_coord_train_client.shape[1], 1, 1))
        ####Image
        X_img_train_client = get_data_files(selected_paths,'image','img')
        X_img_train_client = X_img_train_client/ 255
        ####Lidar
        X_lidar_train_client = get_data_files(selected_paths,'lidar','lidar')


        clients_data[clients] = {'X_lidar_train_client':X_lidar_train_client, 'X_img_train_client':X_img_train_client,'X_coord_train_client':X_coord_train_client,
                                'RF_train_client':y_train_client}
        lens_of_each_client.append(len(RF_train_client))

    # all_together
    for c in experiment_epiosdes:
        try:
            y_train = np.concatenate((y_train,clients_data[c]['RF_train_client']),axis = 0)
            X_lidar_train = np.concatenate((X_lidar_train,clients_data[c]['X_lidar_train_client']),axis = 0)
            X_img_train = np.concatenate((X_img_train,clients_data[c]['X_img_train_client']),axis = 0)
            X_coord_train = np.concatenate((X_coord_train,clients_data[c]['X_coord_train_client']),axis = 0)

        except NameError:
            y_train = clients_data[c]['RF_train_client']
            X_lidar_train = clients_data[c]['X_lidar_train_client']
            X_img_train = clients_data[c]['X_img_train_client']
            X_coord_train = clients_data[c]['X_coord_train_client']

    print('lens_of_each_client',lens_of_each_client)

    print('****************Loading test set*****************')
    RF_test = open_npz('/home/batool/FL/baseline_code/all_test/'+'/'+'rf'+'_'+'all.npz','rf')
    X_lidar_test =  open_npz('/home/batool/FL/baseline_code/all_test/'+'/'+'lidar'+'_'+'all.npz','lidar')
    X_img_test =  open_npz('/home/batool/FL/baseline_code/all_test/'+'/'+'image'+'_'+'all.npz','img')
    X_coord_test =  open_npz('/home/batool/FL/baseline_code/all_test/'+'/'+'gps'+'_'+'all.npz','gps')

    X_coord_test = X_coord_test / 9747
    ## For convolutional input
    X_coord_test = X_coord_test.reshape((X_coord_test.shape[0], X_coord_test.shape[1], 1, 1))
    ####Image
    X_img_test = X_img_test/ 255
    ####Lidar


    print('Check shapes',X_lidar_train.shape,X_img_train.shape,X_coord_train.shape,y_train.shape,X_lidar_test.shape,X_img_test.shape,X_coord_test.shape,RF_test.shape)
    # y_test, _ = custom_label(RF_test,'one_hot')
    y_test  = RF_test
    print('**********train and test set are generated************')
    # y_train = y_train/np.max(y_train)
    # y_test = y_test/np.max(y_test)
    # print('y_train',y_train)
    print('*******HERE*************')
    server = CIFAR10AdaptiveServer(args, config, VGG11())
    list_models, list_indices = server.init_clients()

    # print('samples input',list_indices, MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,num_slices)   #10001 640 False 10
    print('total samples',sum([len(i) for i in list_indices]))
    # sampler = FLSampler(list_indices, MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
    #                     num_slices)
    # print("Sampler initialized")
    # train_loader = get_data_loader(EXP_NAME, X_lidar_train,X_lidar_test,y_train,y_test,data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
    #                                sampler=sampler, num_workers=8, pin_memory=True)

    client_list = [CIFAR10AdaptiveClient(list_models[idx], config, args.use_adaptive) for idx in range(NUM_CLIENTS)]
    print('client_list',client_list)
    # print('client_list',client_list)
    client_id = 0
    for client in client_list:
        client.init_optimizer()
        train_loader = get_data_loader(EXP_NAME, X_lidar_train,X_lidar_test,X_img_train,X_img_test,X_coord_train,X_coord_test,y_train,y_test,client_id,data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,num_workers=8, pin_memory=True)
        client.init_train_loader(train_loader)
        client_id += 1

    print("All initialized. Experiment is {}. Use adaptive = {}. Use initial pruning = {}. Client selection = {}. "
          "Num users = {}. Seed = {}. Max round = {}. "
          "Target density = {}".format(EXP_NAME, args.use_adaptive, args.initial_pruning, args.client_selection,
                                       num_users, args.seed, MAX_ROUND, args.target_density))

    fl_runner = AdaptiveFL(args, config, server, client_list)
    fl_runner.main()
