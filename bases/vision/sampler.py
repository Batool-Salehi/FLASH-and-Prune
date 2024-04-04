from torch.utils.data import Sampler
from copy import deepcopy
from typing import List
from utils.functional import copy_shuffle_list
import random


# class FLSampler(Sampler):
#     def __init__(self, list_indices, num_round, batch_size, num_local_updates, random_client=False, num_client=None):
#         self.sequence = []
#         self.num_round = num_round
#         data_one_round = num_local_updates * batch_size
#         copy_list_ind = deepcopy(list_indices)
#
#         for indices in copy_list_ind:
#             init_indices = deepcopy(indices)  # initial indices
#             while len(indices) < data_one_round * num_round:
#                 indices.extend(copy_shuffle_list(init_indices))
#
#         if random_client:
#             copy_list_ind_pos = [[indices, 0] for indices in copy_list_ind]
#             assert num_client is not None
#             for rd_idx in range(num_round):
#                 # randomly select num_client indices
#                 selected_list_indices = random.sample(copy_list_ind_pos, num_client)
#                 for idx, (indices, pos) in enumerate(selected_list_indices):
#                     self.sequence.extend(indices[pos:pos + data_one_round])
#                     selected_list_indices[idx][1] += data_one_round
#         else:
#             for rd_idx in range(num_round):
#                 for indices in copy_list_ind:
#                     self.sequence.extend(indices[rd_idx * data_one_round: (rd_idx + 1) * data_one_round])
#
#     def __iter__(self):
#         return iter(self.sequence)
#
#     def __len__(self):
#         return len(self.sequence)


# class FLSampler(Sampler):
#     def __init__(self, indices_partition: List[List], num_round, data_per_client, client_selection,
#                  client_per_round=None):
#         self.sequence = []
#         num_partition = len(indices_partition)
#         range_partition = list(range(num_partition))
#         copy_list_ind = deepcopy(indices_partition)
#         new_list_ind = [[] for _ in range(num_partition)]

#         if client_selection:
#             assert client_per_round is not None
#             assert client_per_round <= num_partition

#         list_pos = [0] * num_partition
#         for rd_idx in range(num_round):
#             if client_selection:
#                 selected_client_idx = random.sample(range_partition, client_per_round)
#             else:
#                 selected_client_idx = range_partition

#             for client_idx in selected_client_idx:
#                 ind = copy_list_ind[client_idx]
#                 pos = list_pos[client_idx]
#                 while len(new_list_ind[client_idx]) < pos + data_per_client:
#                     random.shuffle(ind)
#                     new_list_ind[client_idx].extend(ind)
#                 self.sequence.extend(new_list_ind[client_idx][pos:pos + data_per_client])
#                 list_pos[client_idx] = pos + data_per_client

#     def __iter__(self):
#         return iter(self.sequence)

#     def __len__(self):
#         return len(self.sequence)

class FLSampler(Sampler):
    def __init__(self, indices_partition: List[List], num_round, data_per_client, client_selection,
                 client_per_round=None):
        lens_of_each_client = [2910, 3054, 2904, 2887, 2965, 2873, 2706, 2881]#, 2732, 2724]# added
        self.sequence = []
        num_partition = len(indices_partition)
        range_partition = list(range(num_partition))
        copy_list_ind = deepcopy(indices_partition)
        new_list_ind = [[] for _ in range(num_partition)]

        if client_selection:
            assert client_per_round is not None
            assert client_per_round <= num_partition

        list_pos = [0] * num_partition

        for rd_idx in range(num_round):
            if client_selection:
                selected_client_idx = random.sample(range_partition, client_per_round)
            else:
                selected_client_idx = range_partition
            for client_idx in selected_client_idx:
                ind = copy_list_ind[client_idx]
                pos = list_pos[client_idx]
                while len(new_list_ind[client_idx]) < pos + lens_of_each_client[client_idx]:
                    # random.shuffle(ind)
                    new_list_ind[client_idx].extend(ind)
                # print("self.sequence",len(self.sequence))
                self.sequence.extend(new_list_ind[client_idx][pos:pos + lens_of_each_client[client_idx]])
                list_pos[client_idx] = pos + lens_of_each_client[client_idx]
            # if rd_idx>0:
            #     print('Check if equal',self.sequence[0:28636]==self.sequence[rd_idx*28636:(rd_idx+1)*28636])
        print('length of FL sequence.',len(self.sequence))
    def __iter__(self):
        return iter(self.sequence)

    def __len__(self):
        return len(self.sequence)



class FLSampler(Sampler):
    def __init__(self, indices_partition: List[List], num_round, data_per_client, client_selection,
                 client_per_round=None):
        lens_of_each_client = [2910, 3054, 2904, 2887, 2965, 2873, 2706, 2881]#, 2732, 2724]# added
        self.sequence = []
        num_partition = len(indices_partition)
        range_partition = list(range(num_partition))
        copy_list_ind = deepcopy(indices_partition)
        new_list_ind = [[] for _ in range(num_partition)]

        if client_selection:
            assert client_per_round is not None
            assert client_per_round <= num_partition

        list_pos = [0] * num_partition

        for rd_idx in range(1):  # was num_round
            if client_selection:
                selected_client_idx = random.sample(range_partition, client_per_round)
            else:
                selected_client_idx = range_partition
            for client_idx in selected_client_idx:
                ind = copy_list_ind[client_idx]
                pos = list_pos[client_idx]
                while len(new_list_ind[client_idx]) < pos + lens_of_each_client[client_idx]:
                    # random.shuffle(ind)
                    new_list_ind[client_idx].extend(ind)
                # print("self.sequence",len(self.sequence))
                self.sequence.extend(new_list_ind[client_idx][pos:pos + lens_of_each_client[client_idx]])
                list_pos[client_idx] = pos + lens_of_each_client[client_idx]
            # if rd_idx>0:
            #     print('Check if equal',self.sequence[0:28636]==self.sequence[rd_idx*28636:(rd_idx+1)*28636])
        print('length of FL sequence.',len(self.sequence))
        # print('checkrange',all(2910<i< 2910+3054 for i in self.sequence[0:2910]),all(i >= 2910 for i in self.sequence[2910:2910+3054]))
    def __iter__(self):
        return iter(self.sequence)

    def __len__(self):
        return len(self.sequence)
