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


def variable_summaries(var, name):
    with tf.name_scope(name+'/summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        # with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
            # tf.summary.scalar('max', tf.reduce_max(var))
            # tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def inference(images, keep_prob, is_train):
    # conv1
    with tf.variable_scope('conv1'):
        kernel = weight_variable([5, 5, SIMPLE_NUM, 64], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([64], name='biases')
        conv1 = conv2d(images, kernel) + biases
        conv1_bn = batch_norm(conv1, 64, is_train)
        conv1_activation = ACTIVATION(conv1_bn, name='activate')  # 64*64
        conv1_dropout = tf.nn.dropout(conv1_activation, keep_prob=1.0)
        # variable_summaries(conv1)
        # variable_summaries(conv1_bn)
        variable_summaries(conv1_activation, 'conv1_activation')
        variable_summaries(conv1_dropout, 'conv1_dropout')
    # pool1
    with tf.variable_scope('pool1'):
        pool1 = max_pool_2x2(conv1_dropout)  # 32*32

    # conv2
    with tf.variable_scope('conv2'):
        kernel = weight_variable([5, 5, 64, 64], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([64], name='biases')
        conv2 = conv2d(pool1, kernel) + biases
        conv2_bn = batch_norm(conv2, 64, is_train)
        conv2_activation = ACTIVATION(conv2_bn, name='activate')  # 32*32
        conv2_dropout = tf.nn.dropout(conv2_activation, keep_prob=0.7)

        # variable_summaries(conv2)
        # variable_summaries(conv2_bn)
        variable_summaries(conv2_dropout, 'conv2_dropout')
        variable_summaries(conv2_activation, 'conv2_activation')

    # pool2
    with tf.variable_scope('pool2'):
        pool2 = tf.nn.max_pool(conv2_dropout, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 16*16
        h_pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 64])
        h_pool2_flat_drop = tf.nn.dropout(h_pool2_flat, keep_prob)
        variable_summaries(h_pool2_flat, 'h_pool2_flat')

    # fc1
    with tf.variable_scope('fc1'):
        weights = weight_variable([16 * 16 * 64, 512], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([512], name='biases')
        fc_1 = tf.nn.relu(tf.matmul(h_pool2_flat_drop, weights) + biases)
        variable_summaries(fc_1, 'fc_1')
        fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

    # # fc2
    # with tf.variable_scope('fc2'):
    #     weights = weight_variable([500, 500], stddev=0.1, name='weights', wd=0.01)
    #     biases = bias_variable([500], name='biases')
    #     fc_2 = tf.nn.relu(tf.matmul(fc_1, weights) + biases)
    #     variable_summaries(fc_2)
    # fc2_drop = tf.nn.dropout(fc_2, keep_prob)

    # fc2  facial point
    with tf.variable_scope('fc4_fp'):
        weights = weight_variable([16 * 16 * 64, LANDMARKS_LENGTH], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([LANDMARKS_LENGTH], name='biases')
        fp_logits = tf.matmul(h_pool2_flat_drop, weights) + biases

    # fc3 facial expression
    with tf.variable_scope('fc3_ep'):
        weights = weight_variable([512, NUM_CLASSES], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([NUM_CLASSES], name='biases')
        fe_logits = tf.matmul(fc_1_drop, weights) + biases

    return fp_logits, fe_logits