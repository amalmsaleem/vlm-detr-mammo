import torch
from torch import nn
from torch.nn import functional as F

from typing import Dict, List
from util.misc import NestedTensor, is_main_process


from models.backbone_EN_utils import (
    relu_fn,
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (
                0 < self._block_args.se_ratio <= 1
        )
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = (
                self._block_args.input_filters * self._block_args.expand_ratio
        )  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False
            )
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
            )

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=k,
            stride=s,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
        )

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio)
            )
            self._se_reduce = Conv2d(
                in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1
            )
            self._se_expand = Conv2d(
                in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1
            )

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps
        )

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = (
            self._block_args.input_filters,
            self._block_args.output_filters,
        )
        if (
                self.id_skip
                and self._block_args.stride == 1
                and input_filters == output_filters
        ):
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        if not isinstance(blocks_args, list):
            raise AssertionError("blocks_args should be a list")
        if len(blocks_args) <= 0:
            raise AssertionError("block args must be greater than 0")
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(
            32, self._global_params
        )  # number of output channels
        self._conv_stem = Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=False
        )
        self._bn0 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(
                    block_args.input_filters, self._global_params
                ),
                output_filters=round_filters(
                    block_args.output_filters, self._global_params
                ),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params),
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )

        # Final linear layer
        self._dropout = self._global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        self.source_layer_indexes = []

    # def extract_features(self, inputs):
    #     """Returns output of the final convolution layer"""

    #     # Stem
    #     ###count 21, 29
    #     x = relu_fn(self._bn0(self._conv_stem(inputs)))
    #     # print ("Extract_features fun:: begining", x.shape)

    #     # Blocks
    #     flag = 0
    #     pointer = 0
    #     for idx, block in enumerate(self._blocks):
    #         drop_connect_rate = self._global_params.drop_connect_rate
    #         if drop_connect_rate:
    #             drop_connect_rate *= float(idx) / len(self._blocks)
    #         x = block(x, drop_connect_rate=drop_connect_rate)
    #         # print ("Extract_features fun:: for loop count::", idx , x.shape)
    #         if idx == self.source_layer_indexes[pointer] and flag == 0:
    #             C3 = x
    #             flag = 1
    #             pointer += 1
    #         if idx == self.source_layer_indexes[pointer]:
    #             C4 = x
    #         # if idx==31:
    #         #  C5 = x

    #     # Head
    #     x = relu_fn(self._bn1(self._conv_head(x)))
    #     # print ("Extract_features fun:: ending", x.shape)

    #     # return x
    #     return x, C3, C4

    # def forward(self, inputs):
    #     """Calls extract_features to extract features, and returns features that are used for constructing FPN network."""
    #     x, C3, C4 = self.extract_features(inputs)
    #     C5 = x
    #     return (x, C3, C4, C5)
    
    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])
                >>> print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        return endpoints

    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        #amal _swish to relu_fn
        # x = relu_fn(self._bn0(self._conv_stem(inputs)))

        # # Blocks
        # for idx, block in enumerate(self._blocks):
        #     drop_connect_rate = self._global_params.drop_connect_rate
        #     if drop_connect_rate:
        #         drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
        #     x = block(x, drop_connect_rate=drop_connect_rate)

        # # Head
        # x = relu_fn(self._bn1(self._conv_head(x)))
        # # x = relu_fn(self._bn1(self._conv_head(x)))

        # x = relu_fn(self._bn0(self._conv_stem(inputs)))
        # features_dict = {}
        # # print ("Extract_features fun:: begining", x.shape)

        # # Blocks
        # flag = 0
        # pointer = 0
        # for idx, block in enumerate(self._blocks):
        #     drop_connect_rate = self._global_params.drop_connect_rate
        #     if drop_connect_rate:
        #         drop_connect_rate *= float(idx) / len(self._blocks)
        #     x = block(x, drop_connect_rate=drop_connect_rate)
        #     # print ("Extract_features fun:: for loop count::", idx , x.shape)
        #     if idx == self.source_layer_indexes[pointer] and flag == 0:
        #         # C3 = x
        #         features_dict['C3'] = x 
        #         flag = 1
        #         pointer += 1
        #     if idx == self.source_layer_indexes[pointer]:
        #         # C4 = x
        #         features_dict['C4'] = x
            # if idx==31:
            #  C5 = x

        # Head
        # x = relu_fn(self._bn1(self._conv_head(x)))
        # features_dict['output'] = x
        # print ("Extract_features fun:: ending", x.shape)

    #     x = relu_fn(self._bn0(self._conv_stem(inputs)))

    # # Intermediate feature layers
    #     C3, C4, C5 = None, None, None

    #     # Blocks
    #     for idx, block in enumerate(self._blocks):
    #         drop_connect_rate = self._global_params.drop_connect_rate
    #         if drop_connect_rate:
    #             drop_connect_rate *= float(idx) / len(self._blocks)
    #         x = block(x, drop_connect_rate=drop_connect_rate)

    #         # Capture intermediate features similar to ResNet's layer2, layer3, and layer4
    #         if idx == 6:     # layer2-like output
    #             C3 = x
    #         elif idx == 13:  # layer3-like output
    #             C4 = x
    #         elif idx == 26:  # layer4-like output
    #             C5 = x

    #     # Head
    #     x = relu_fn(self._bn1(self._conv_head(x)))

    #     return {'2': C3, '3': C4, '4': C5, 'x': x}
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

            # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = relu_fn(self._bn1(self._conv_head(x)))
        # print("x: ",x.shape)

        return {'0' : x}

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        # print(f"Input type: {type(inputs)}")            # Check the type of inputs
        # print(f"Input shape: {inputs.tensors.shape}")   # Shape of the tensors if NestedTensor
        # if hasattr(inputs, 'mask'):
        #     print(f"Input mask shape: {inputs.mask.shape}") 

        xs = self.extract_features(inputs.tensors)
        # print(f"xs type: {type(xs)}")            # Check the type of inputs
        # print(f"xs shape: {xs.shape}") 
        # Pooling and final linear layer
        # x = self._avg_pooling(x)
        # if self._global_params.include_top:
        #     x = x.flatten(start_dim=1)
        #     x = self._dropout(x)
        #     x = self._fc(x)

        
#amal
        # xs = self.body(x.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = inputs.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            # print(f"mask shape: {mask.shape}") 
            # print(f"x shape: {x.shape}") 
            out[name] = NestedTensor(x, mask)
        return out
        # return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        print(global_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000, model_type=None, clip_chk_pt=None,
                        freeze_backbone="n"):
        model = EfficientNet.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        load_pretrained_weights(
            model, model_name, model_type, load_fc=(num_classes == 1000), clip_chk_pt=clip_chk_pt,
            freeze_backbone=freeze_backbone)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment."""
        num_models = 4 if also_need_pretrained_weights else 10
        valid_models = ["efficientnet_b" + str(i) for i in range(num_models)]
        print(model_name)
        if model_name.replace("-", "_") not in valid_models:
            raise ValueError("model_name should be one of: " + ", ".join(valid_models))
