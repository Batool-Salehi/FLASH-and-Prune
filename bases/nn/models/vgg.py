from torch import nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits
import torch
from bases.nn.conv2d import DenseConv2d
from bases.nn.linear import DenseLinear
from bases.nn.models.base_model import BaseModel
from bases.nn.sequential import DenseSequential
from torch.nn import CrossEntropyLoss

__all__ = ["VGG11"]

class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class VGG11(BaseModel):
    def __init__(self, dict_module: dict = None):
        if dict_module is None:
            dict_module = dict()
            self.batch_norm = False
            self.config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

            features_lidar = self._make_feature_layers_lidar()
            features_img = self._make_feature_layers_img()
            features_gps = self._make_feature_layers_gps()

            classifier = DenseSequential(DenseLinear(832, 2048, a=0),  #800, 3200
                                        nn.ReLU(inplace=True),
                                        # nn.BatchNorm1d(2048),

                                        DenseLinear(2048, 1024, a=0),
                                        nn.ReLU(inplace=True),
                                        # nn.BatchNorm1d(1024),

                                        DenseLinear(1024, 512, a=0),
                                        nn.ReLU(inplace=True),
                                        # nn.BatchNorm1d(512),

                                        DenseLinear(512, 256, a=0),
                                        nn.ReLU(inplace=True),
                                        # nn.BatchNorm1d(256),

                                        DenseLinear(256, 128, a=0),
                                        nn.ReLU(inplace=True),
                                        # nn.BatchNorm1d(128),

                                        DenseLinear(128, 64, a=1.5, mode="fan_out")
                                        # ,nn.Softmax(dim=1)
                                        )

            dict_module["features_lidar"] = features_lidar
            dict_module["features_img"] = features_img
            dict_module["features_gps"] = features_gps

            dict_module["classifier"] = classifier

        super(VGG11, self).__init__(CrossEntropyLoss, dict_module)

    def collect_layers(self):
        self.get_param_layers(self.param_layers, self.param_layer_prefixes)
        prunable_nums = [ly_id for ly_id, ly in enumerate(self.param_layers) if not isinstance(ly, nn.BatchNorm2d)]
        self.prunable_layers = [self.param_layers[ly_id] for ly_id in prunable_nums]
        self.prunable_layer_prefixes = [self.param_layer_prefixes[ly_id] for ly_id in prunable_nums]

    def _make_feature_layers_lidar(self):
        layers = []
        in_channels = 20
        print('********Model is intialized*************')

        return nn.Sequential(

            DenseConv2d(in_channels, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),

            ResNet(
                nn.Sequential(
                    DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                    DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)
                    )),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.3),

            ResNet(
                nn.Sequential(
                    DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                    DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)
                    )),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.3),

            ResNet(
                nn.Sequential(
                    DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                    DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)
                    )),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(p=0.3),

            ResNet(
                nn.Sequential(
                    DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                    DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                    )),
            nn.Flatten(),
            DenseLinear(320, 1024, a=0),
            nn.Dropout(p=0.2),
            DenseLinear(1024, 512, a=0),
            nn.Dropout(p=0.2)
                    )


    def _make_feature_layers_img(self):
        layers = []
        in_channels = 90  ###
        print('********Model is intialized*************')
        return nn.Sequential(
            DenseConv2d(in_channels, 32, kernel_size=7, padding=3),nn.ReLU(inplace=True),

            ResNet(
                nn.Sequential(
                    DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                    DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                    DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)

                    )),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),

            ResNet(
                nn.Sequential(
                    DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                    DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)
                    )),
            nn.MaxPool2d((3, 3), padding=1),
            nn.Dropout(p=0.25),
            nn.Flatten(),
            DenseLinear(864, 512, a=0),nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            DenseLinear(512, 256, a=0),nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            DenseLinear(256, 256, a=0),nn.Tanh()
            )

    def _make_feature_layers_gps(self):
        layers = []
        in_channels = 2
        print('********Model is intialized*************')
        return nn.Sequential(
            DenseConv2d(in_channels, 20, kernel_size=2, padding=1),nn.ReLU(inplace=True),
            DenseConv2d(20, 20, kernel_size=2, padding=1),nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            DenseLinear(20, 1024, a=0),nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            DenseLinear(1024, 512, a=0),nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            DenseLinear(512, 256, a=0),nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            DenseLinear(256, 64, a=0),nn.ReLU(inplace=True),nn.Tanh()
            )


    def forward(self, inputs1,inputs2,inputs3):
        # print('check point',inputs.shape)
        outputs_lidar = self.features_lidar(inputs1)
        # print('outputs_lidar features',outputs_lidar.shape)
        outputs_lidar = outputs_lidar.view(outputs_lidar.size(0), -1)

        outputs_img = self.features_img(inputs2)
        # print('outputs_img features',outputs_img.shape)
        outputs_img = outputs_img.view(outputs_img.size(0), -1)

        outputs_gps = self.features_gps(inputs3)
        # print('outputs_gps features',outputs_gps.shape)
        outputs_gps = outputs_gps.view(outputs_gps.size(0), -1)

        outputs = torch.cat((outputs_lidar, outputs_img, outputs_gps), dim=1)
        # print('check point after concatination',outputs.shape)
        outputs = outputs.view(outputs.size(0), -1) #old
        # outputs = outputs.view(-1,32*5*5)   # outputs 64 only
        # print('outputs flatten',outputs.shape)
        outputs = self.classifier(outputs)
        # print('classifier output shape',outputs)
        return outputs

    def to_sparse(self):
        new_features = [ft.to_sparse() if isinstance(ft, DenseConv2d) else ft for ft in self.features]
        new_module_dict = {"features": nn.Sequential(*new_features), "classifier": self.classifier.to_sparse()}
        return self.__class__(new_module_dict)







    # old
    # def _make_feature_layers(self):
    #     layers = []
    #     in_channels = 20
    #     print('********Model is intialized*************')
    #     layers.extend([DenseConv2d(in_channels, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)])
    #     layers.extend([DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)])
    #     layers.extend([DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)])
    #     layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    #     layers.append(nn.Dropout(p=0.3, inplace=True))

    #     layers.extend([DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)])
    #     layers.extend([DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)])
    #     layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    #     layers.append(nn.Dropout(p=0.3, inplace=True))

    #     layers.extend([DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)])
    #     layers.extend([DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)])
    #     layers.append(nn.MaxPool2d(kernel_size=1, stride=1))
    #     layers.append(nn.Dropout(p=0.3, inplace=True))

    #     layers.extend([DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)])
    #     layers.extend([DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)])


    # def _make_feature_layers_lidar(self):
    #     layers = []
    #     in_channels = 20
    #     print('********Model is intialized*************')
    #     layers.extend([DenseConv2d(in_channels, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)])
    #     layers.extend([DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)])
    #     layers.extend([DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)])
    #     layers.append(nn.MaxPool2d(kernel_size=2))
    #     layers.append(nn.Dropout(p=0.3, inplace=True))

    #     layers.extend([DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)])
    #     layers.extend([DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)])
    #     layers.append(nn.MaxPool2d(kernel_size=2))
    #     layers.append(nn.Dropout(p=0.3, inplace=True))

    #     # layers.extend([DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)])
    #     # layers.extend([DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)])
    #     # layers.append(nn.MaxPool2d((1, 2)))
    #     # layers.append(nn.Dropout(p=0.3, inplace=True))
    #     # layers.append(DenseLinear(800, 1024, a=0),nn.ReLU(inplace=True))
    #     # layers.append(nn.Dropout(p=0.2, inplace=False))
    #     # layers.append(DenseLinear(1024, 256, a=0),nn.ReLU(inplace=True))
    #     # layers.append(nn.Dropout(p=0.2, inplace=False))
    #     # layers.append(DenseLinear(256, 64, a=1.5)

    #     return nn.Sequential(*layers)



    # def _make_feature_layers_lidar(self):
    #     layers = []
    #     in_channels = 20
    #     print('********Model is intialized*************')

    #     return nn.Sequential(

    #         DenseConv2d(in_channels, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
    #         DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
    #         DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)
    #         # ResNet(
    #         #     nn.Sequential(
    #         #         DenseConv2d(in_channels, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)
    #         #         # DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
    #         #         # DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)
    #         #         )),

    #         # nn.MaxPool2d(kernel_size=2),
    #         # nn.Dropout(p=0.3, inplace=True),
    #         # DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
    #         # DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
    #         # DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
    #         # nn.Dropout(p=0.3, inplace=True)
    #         )
    #     # return nn.Sequential(*layers)
