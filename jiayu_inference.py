import tensorflow as tf
import math

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
    # tf.layers.batch_normalization()
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
    with tf.variable_scope('ds0'):
        kernel = weight_variable([5, 5, SIMPLE_NUM, 64], stddev=0.1, name='weights', wd=0.01)
        # biases = bias_variable([64], name='biases')
        ds0 = conv2d(images, kernel) #+ biases
        # ds0_bn = batch_norm(ds0, 64, is_train)
        # ds0_activation = ACTIVATION(ds0_bn, name='activate')  # 64*64
        # conv1_dropout = tf.nn.dropout(conv1_activation, keep_prob=1.0)
        variable_summaries(ds0, 'ds0')

        ds1_input = tf.concat([images, ds0], 3)
        pool1 = max_pool_2x2(ds1_input)  # 32*32

    with tf.variable_scope('ds1'):
        ds1_bn = batch_norm(pool1, SIMPLE_NUM+64, is_train)
        ds0_activation = ACTIVATION(ds1_bn, name='activate')  # 64*64
        kernel = weight_variable([5, 5, SIMPLE_NUM+64, 64], stddev=0.1, name='weights', wd=0.01)
        # biases = bias_variable([64], name='biases')
        ds1 = conv2d(ds0_activation, kernel) #+ biases

        # conv2_dropout = tf.nn.dropout(conv2_activation, keep_prob=1.0)
        variable_summaries(ds1, 'ds1')

        ds2_input = tf.concat([pool1, ds1], 3)
        pool2 = max_pool_2x2(ds2_input)  # 16*16

    with tf.variable_scope('ds2'):
        ds2_bn = batch_norm(pool2, SIMPLE_NUM + 64*2, is_train)
        ds2_activation = ACTIVATION(ds2_bn, name='activate')  # 32*32
        kernel = weight_variable([3, 3, SIMPLE_NUM + 64*2, 64], stddev=0.1, name='weights', wd=0.01)
        # biases = bias_variable([64], name='biases')
        ds2 = conv2d(ds2_activation, kernel) #+ biases

        variable_summaries(ds2, 'ds2')
        ds3_input = tf.concat([pool2, ds2], 3)
        pool3 = max_pool_2x2(ds3_input)  # 8*8

    with tf.variable_scope('ds3'):
        ds3_bn = batch_norm(pool3, SIMPLE_NUM + 64*3, is_train)
        ds3_activation = ACTIVATION(ds3_bn, name='activate')  # 16*16
        kernel = weight_variable([3, 3, SIMPLE_NUM + 64*3, 64], stddev=0.1, name='weights', wd=0.01)
        # biases = bias_variable([64], name='biases')
        ds3 = conv2d(ds3_activation, kernel) # + biases

        # conv2_dropout = tf.nn.dropout(conv2_activation, keep_prob=1.0)
        variable_summaries(ds3, 'ds3')
        ds4_input = tf.concat([pool3, ds3], 3)
        pool4 = max_pool_2x2(ds4_input)  # 4*4   7+64*4

    pool4_flat = tf.reshape(pool4, [-1, 4*4*(7+64*4)])

    # fc facial expression
    with tf.variable_scope('fc1'):
        weights = weight_variable([4*4*(7+64*4), NUM_CLASSES], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([NUM_CLASSES], name='biases')
        logits = tf.matmul(pool4_flat, weights) + biases

    return logits


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
                                    0.6,  # 0.96,
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