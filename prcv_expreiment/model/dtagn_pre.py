import tensorflow as tf


IMAGE_SIZE = 64
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
CK_NUM_CLASSES = 8
CK_SIMPLE_NUM = 6
CK_LANDMARKS_LENGTH = 68*2*CK_SIMPLE_NUM

OULU_NUM_CLASSES = 6
OULU_SIMPLE_NUM = 7
OULU_LANDMARKS_LENGTH = 68*2*OULU_SIMPLE_NUM
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
    with tf.variable_scope('dtgn_bn'):
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


def variable_summaries(var, name, is_conv=False):
    with tf.name_scope(name + '/summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        # with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
            # tf.summary.scalar('max', tf.reduce_max(var))
            # tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
        if is_conv:
            tf.summary.image('image', tf.reshape(var[:, :, :, 0], [-1, 64, 64, 1]))


def inference(images, landmarks, keep_prob, is_train):
    # conv1
    with tf.variable_scope('dtan_conv1'):
        kernel = weight_variable([5, 5, OULU_SIMPLE_NUM, 64], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([64], name='biases')
        conv1 = conv2d(images, kernel) + biases
        conv1_bn = batch_norm(conv1, 64, is_train)
        conv1_activation = ACTIVATION(conv1_bn, name='activate')  # 64*64
        # variable_summaries(conv1)
        # variable_summaries(conv1_bn)
        # variable_summaries(conv1_activation, "conv1")
    # pool1
    with tf.variable_scope('dtan_pool1'):
        pool1 = max_pool_2x2(conv1_activation)  # 32*32

    # conv2
    with tf.variable_scope('dtan_conv2'):
        kernel = weight_variable([5, 5, 64, 64], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([64], name='biases')
        conv2 = conv2d(pool1, kernel) + biases
        conv2_bn = batch_norm(conv2, 64, is_train)
        conv2_activation = ACTIVATION(conv2_bn, name='activate')  # 64*64
        # variable_summaries(conv2)
        # variable_summaries(conv2_bn)
        variable_summaries(conv2_activation, "conv2")

    # pool2
    with tf.variable_scope('dtan_pool2'):
        pool2 = tf.nn.max_pool(conv2_activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 16*16

    # fc1
    h_pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 64])
    with tf.variable_scope('dtan_fc1'):
        weights = weight_variable([16 * 16 * 64, 500], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([500], name='biases')
        fc_1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights) + biases)
        variable_summaries(fc_1, 'fc1')
        # fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

    with tf.variable_scope('dtgn_features'):
        weights = weight_variable([16 * 16 * 64, 600], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([600], name='biases')
        dtgn_features = tf.nn.relu(tf.matmul(h_pool2_flat, weights) + biases)
        variable_summaries(fc_1, 'fc1')
        # fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

    # fc2
    with tf.variable_scope('dtan_fc2'):
        weights = weight_variable([500, 500], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([500], name='biases')
        fc_2 = tf.nn.relu(tf.matmul(fc_1, weights) + biases)
        variable_summaries(fc_2, 'fc2')
        fc2_drop = tf.nn.dropout(fc_2, keep_prob)

    # fc3 facial expression
    with tf.variable_scope('dtan_fc3_ep'):
        weights = weight_variable([500, OULU_NUM_CLASSES], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([OULU_NUM_CLASSES], name='biases')
        fe_logits = tf.matmul(fc2_drop, weights) + biases


    """
    dtgn network
    """
    with tf.variable_scope('dtgn_fc1'):
        weights = weight_variable([OULU_LANDMARKS_LENGTH, 100], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([100], name='biases')
        fc_1 = tf.nn.relu(tf.matmul(landmarks, weights) + biases)
        variable_summaries(fc_1, 'fc1')
        # fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

    # fc2
    with tf.variable_scope('dtgn_fc2'):
        weights = weight_variable([100, 600], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([600], name='biases')
        dtgn_fc2 = tf.nn.relu(tf.matmul(fc_1, weights) + biases)
        variable_summaries(fc_2, 'fc2')

        return_weights = weights
        # fc2_drop = tf.nn.dropout(fc_2, keep_prob)

    return fe_logits, dtgn_features, dtgn_fc2, return_weights


def loss(logits, labels_placeholder, dtgn_features, dtgn_fc2):
    squre_error = tf.reduce_sum(tf.pow(dtgn_features - dtgn_fc2, 2)) / 600
    squre_error_mean = tf.reduce_mean(squre_error, name='squre_error_mean')

    labels = tf.to_int64(labels_placeholder)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    xentropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', squre_error_mean * 0.8 + xentropy_mean * 0.2)
    # tf.add_to_collection('losses', xentropy_mean)
    tf.summary.scalar('xentropy_mean', xentropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def training(total_loss, init_learning_rate, global_step, tra_vars):
    lr = tf.train.exponential_decay(init_learning_rate,
                                    global_step,
                                    2500 ,
                                    0.1,  # 0.96  0.3
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(total_loss, global_step=global_step, var_list=tra_vars)
    return train_op


def evaluation(logits, labels):
    logits = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy