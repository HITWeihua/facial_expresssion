import tensorflow as tf

IMAGE_SIZE = 64
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 6
SIMPLE_NUM = 7
LANDMARKS_LENGTH = 68*2*SIMPLE_NUM
ACTIVATION = tf.nn.relu


def batch_norm(x, n_out, is_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def weight_variable(shape, stddev, name, wd):
    initial = tf.truncated_normal(shape, stddev=stddev, name=name)
    var = tf.Variable(initial)
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        # tf.summary.histogram('histogram', var)


def res_block(input_layer, is_train, channels):
    kernel1 = weight_variable([5, 5, channels, channels], stddev=0.1, name='weights', wd=0.0)
    biases1 = bias_variable([channels], name='biases')
    conv1 = conv2d(input_layer, kernel1) + biases1
    conv1_bn = batch_norm(conv1, channels, is_train)
    conv1_activation = ACTIVATION(conv1_bn, name='activate')  # 64*64

    kernel2 = weight_variable([5, 5, channels, channels], stddev=0.1, name='weights', wd=0.0)
    biases2 = bias_variable([channels], name='biases')
    conv2 = conv2d(conv1_activation, kernel2) + biases2

    add_layer = tf.add(conv2, input_layer)

    add_layer_bn = batch_norm(add_layer, channels, is_train)
    add_layer_activation = ACTIVATION(add_layer_bn, name='activate')  # 64*64
    variable_summaries(add_layer_activation)
    return add_layer_activation


def block_top(image, is_train):
    image = tf.reshape(image, [-1, 64, 64, 1])
    kernel1 = weight_variable([5, 5, 1, 16], stddev=0.1, name='weights', wd=0.0)
    biases1 = bias_variable([16], name='biases')
    conv1 = conv2d(image, kernel1) + biases1
    conv1_bn = batch_norm(conv1, 16, is_train)
    conv1_activation = ACTIVATION(conv1_bn, name='activate')  # 64*64

    with tf.variable_scope('pool1'):
        pool1 = max_pool_2x2(conv1_activation)  # 32*32

    with tf.variable_scope("block1"):
        layer_activation1 = res_block(pool1, is_train, 16)  # 32*32

    with tf.variable_scope('pool2'):
        pool2 = max_pool_2x2(layer_activation1)  # 16*16

    with tf.variable_scope('block2'):
        layer_activation2 = res_block(pool2, is_train, 16) # 16*16

    with tf.variable_scope('pool3'):
        pool3 = max_pool_2x2(layer_activation2)  # 8*8

    with tf.variable_scope('block3'):
        layer_activation3 = res_block(pool3, is_train, 16)  # 8*8
    return layer_activation3


def inference(images, keep_prob, is_train):
    with tf.variable_scope('block_top'):
        for i in range(SIMPLE_NUM):
            if i == 0:
                frames_features = block_top(images[:, :, :, i], is_train)
                old_frames_features = frames_features
            elif i == 1:
                frames_features = block_top(images[:, :, :, i], is_train)
                inner_features_concat = frames_features - old_frames_features
                old_frames_features = frames_features
            else:
                frames_features = block_top(images[:, :, :, i], is_train)
                inner_features_concat = tf.concat([inner_features_concat, frames_features - old_frames_features], axis=-1)
                old_frames_features = frames_features

    with tf.variable_scope('block_neck'):
        kernel1 = weight_variable([5, 5, 96, 64], stddev=0.1, name='weights', wd=0.0)
        biases1 = bias_variable([64], name='biases')
        conv1 = conv2d(inner_features_concat, kernel1) + biases1
        conv1_bn = batch_norm(conv1, 64, is_train)
        conv1_activation = ACTIVATION(conv1_bn, name='activate')

    with tf.variable_scope('block4'):
        layer_activation4 = res_block(conv1_activation, is_train, 64)

    with tf.variable_scope('pool4'):
        pool4 = tf.nn.max_pool(layer_activation4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # with tf.variable_scope('block5'):
    #     kernel7 = weight_variable([5, 5, 64, 64], stddev=0.1, name='weights', wd=0.0)
    #     biases7 = bias_variable([64], name='biases')
    #     conv7 = conv2d(pool3, kernel7) + biases7
    #     conv7_bn = batch_norm(conv7, 64, is_train)
    #     conv7_activation = ACTIVATION(conv7_bn, name='activate')  # 8*8
    #
    #     kernel8 = weight_variable([5, 5, 64, 64], stddev=0.1, name='weights', wd=0.0)
    #     biases8 = bias_variable([64], name='biases')
    #     conv8 = conv2d(conv7_activation, kernel8) + biases8
    #
    #     add_layer4 = tf.add(conv8, pool3)
    #
    #     add_layer4_bn = batch_norm(add_layer4, 64, is_train)
    #     add_layer4_activation = ACTIVATION(add_layer4_bn, name='activate')  # 8*8
    #     variable_summaries(add_layer4_activation)
    #
    # # pool2
    # with tf.variable_scope('pool4'):
    #     pool4 = tf.nn.max_pool(add_layer4_activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 4*4

    # fc1
    h_pool4_flat = tf.reshape(pool4, [-1, 4 * 4 * 64])
    with tf.variable_scope('fc1'):
       weights = weight_variable([4 * 4 * 64, 512], stddev=0.1, name='weights', wd=0.01)
       biases = bias_variable([512], name='biases')
       fc_1 = tf.nn.relu(tf.matmul(h_pool4_flat, weights) + biases)
       variable_summaries(fc_1)
       fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

    # fc3 facial expression
    with tf.variable_scope('fc3_ep'):
        weights = weight_variable([512, NUM_CLASSES], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([NUM_CLASSES], name='biases')
        fe_logits = tf.matmul(fc_1_drop, weights) + biases

    return fe_logits


def loss(logits, labels_placeholder):
    labels = tf.to_int64(labels_placeholder)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    xentropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', xentropy_mean)
    tf.summary.scalar('xentropy_mean', xentropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def training(total_loss, init_learning_rate, global_step):
    lr = tf.train.exponential_decay(init_learning_rate,
                                    global_step,
                                    1000,
                                    0.3,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(total_loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    logits = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy
