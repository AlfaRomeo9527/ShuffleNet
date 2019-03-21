import tensorflow as tf

# 构建的shufflenet unit
def shufflenet_unit(name, x, w=None, num_groups=1, group_conv_bottleneck=True, num_filters=16, stride=(1, 1),
                    l2_strength=0.0, bias=0.0, batchnorm_enabled=True, is_training=True, fusion='add'):
    # Paper parameters. If you want to change them feel free to pass them as method parameters.
    activation = tf.nn.relu

    with tf.variable_scope(name) as scope:
        residual = x
        bottleneck_filters = (num_filters // 4) if fusion == 'add' else (num_filters - residual.get_shape()[
            3].value) // 4

        if group_conv_bottleneck:
            # 先1x1卷积
            bottleneck = grouped_conv2d('Gbottleneck', x=x, w=None, num_filters=bottleneck_filters, kernel_size=(1, 1),
                                        padding='VALID',
                                        num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                        activation=activation,
                                        batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            # 通道混洗
            shuffled = channel_shuffle('channel_shuffle', bottleneck, num_groups)
        else:
            bottleneck = conv2d('bottleneck', x=x, w=None, num_filters=bottleneck_filters, kernel_size=(1, 1),
                                padding='VALID', l2_strength=l2_strength, bias=bias, activation=activation,
                                batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            shuffled = bottleneck
        # 3x3深度分离卷积
        padded = tf.pad(shuffled, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        depthwise = depthwise_conv2d('depthwise', x=padded, w=None, stride=stride, l2_strength=l2_strength,
                                     padding='VALID', bias=bias,
                                     activation=None, batchnorm_enabled=batchnorm_enabled, is_training=is_training)
        # 如果步长为2，则下采样
        if stride == (2, 2):
            residual_pooled = avg_pool_2d(residual, size=(3, 3), stride=stride, padding='SAME')
        else:
            residual_pooled = residual

        # 如果是通道连接
        if fusion == 'concat':
            group_conv1x1 = grouped_conv2d('Gconv1x1', x=depthwise, w=None,
                                           num_filters=num_filters - residual.get_shape()[3].value,
                                           kernel_size=(1, 1),
                                           padding='VALID',
                                           num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                           activation=None,
                                           batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            return activation(tf.concat([residual_pooled, group_conv1x1], axis=-1))
        # 最后像素加
        elif fusion == 'add':
            group_conv1x1 = grouped_conv2d('Gconv1x1', x=depthwise, w=None,
                                           num_filters=num_filters,
                                           kernel_size=(1, 1),
                                           padding='VALID',
                                           num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                           activation=None,
                                           batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            residual_match = residual_pooled
            # This is used if the number of filters of the residual block is different from that
            # of the group convolution.
            if num_filters != residual_pooled.get_shape()[3].value:
                residual_match = conv2d('residual_match', x=residual_pooled, w=None, num_filters=num_filters,
                                        kernel_size=(1, 1),
                                        padding='VALID', l2_strength=l2_strength, bias=bias, activation=None,
                                        batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            return activation(group_conv1x1 + residual_match)
        else:
            raise ValueError("Specify whether the fusion is \'concat\' or \'add\'")


# 通道混洗
def channel_shuffle(name, x, num_groups):
    with tf.variable_scope(name) as scope:
        n, h, w, c = x.shape.as_list()
        x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])  # 先合并重组
        x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])  # 转置
        output = tf.reshape(x_transposed, [-1, h, w, c])  # 摊平
        return output
