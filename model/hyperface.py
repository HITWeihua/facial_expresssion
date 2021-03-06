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
        # with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
            # tf.summary.scalar('max', tf.reduce_max(var))
            # tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def inference(images, keep_prob, is_train):
    # conv1
    with tf.variable_scope('conv1'):
        kernel = weight_variable([5, 5, SIMPLE_NUM, 64], stddev=0.1, name='weights', wd=0.0)
        biases = bias_variable([64], name='biases')
        conv1 = conv2d(images, kernel) + biases
        conv1_bn = batch_norm(conv1, 64, is_train)
        conv1_activation = ACTIVATION(conv1_bn, name='activate')  # 64*64
        variable_summaries(conv1)
        variable_summaries(conv1_bn)
        variable_summaries(conv1_activation)
    # pool1
    pool1 = max_pool_2x2(conv1_activation)  # 32*32

    # conv2
    with tf.variable_scope('conv2'):
        kernel = weight_variable([5, 5, 64, 64], stddev=0.1, name='weights', wd=0.0)
        biases = bias_variable([64], name='biases')
        conv2 = conv2d(pool1, kernel) + biases
        conv2_bn = batch_norm(conv2, 64, is_train)
        conv2_activation = ACTIVATION(conv2_bn, name='activate')  # 64*64
        variable_summaries(conv2)
        variable_summaries(conv2_bn)
        variable_summaries(conv2_activation)

    # pool2
    pool2 = tf.nn.max_pool(conv2_activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 16*16

    # conv3
    with tf.variable_scope('conv3'):
        kernal = weight_variable([5, 5, 64, 64], stddev=0.1, name='weights', wd=0.0)
        biases = bias_variable([64], name='biases')
        conv3 = conv2d(pool2, kernal) + biases
        conv3_bn = batch_norm(conv3, 64, is_train)
        conv3_activation = ACTIVATION(conv3_bn, name='activate')  # 64*64
        variable_summaries(conv3)
        variable_summaries(conv3_bn)
        variable_summaries(conv3_activation)

    pool3 = tf.nn.max_pool(conv3_activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 8*8

    # conv4
    with tf.variable_scope('conv4'):
        kernal = weight_variable([4, 4, 64, 64], stddev=0.1, name='weights', wd=0.0)
        biases = bias_variable([64], name='biases')
        conv4 = tf.nn.conv2d(pool1, kernal, strides=[1, 4, 4, 1], padding='SAME') + biases
        conv4_bn = batch_norm(conv4, 64, is_train)
        conv4_activation = ACTIVATION(conv4_bn, name='activate')  # 64*64
        variable_summaries(conv4)
        variable_summaries(conv4_bn)
        variable_summaries(conv4_activation)

    concat = tf.concat([pool3, conv4_activation], 3, name='concat')  # 8*8*128
    # fc_all
    concat_flat = tf.reshape(concat, [-1, 8 * 8 * 128])
    with tf.variable_scope('fc_all'):
        weights = weight_variable([8192, 4096], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([4096], name='biases')
        fc_all = tf.nn.relu(tf.matmul(concat_flat, weights) + biases)
        variable_summaries(fc_all)
    fc_all_drop = tf.nn.dropout(fc_all, keep_prob)
    # fc1
    with tf.variable_scope('fc1'):
        weights = weight_variable([4096, 512], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([512], name='biases')
        fc1 = tf.nn.relu(tf.matmul(fc_all_drop, weights) + biases)
        variable_summaries(fc1)
    fc1_drop = tf.nn.dropout(fc1, keep_prob)
    # fc2  facial point
    with tf.variable_scope('fc2'):
        weights = weight_variable([4096, LANDMARKS_LENGTH], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([LANDMARKS_LENGTH], name='biases')
        fp_logits = tf.matmul(fc_all_drop, weights) + biases

    # fc3 facial expression
    with tf.variable_scope('fc3'):
        weights = weight_variable([512, NUM_CLASSES], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([NUM_CLASSES], name='biases')
        fe_logits = tf.matmul(fc1_drop, weights) + biases

    return fp_logits, fe_logits


def loss(logits, labels_placeholder, landmarks_placeholder):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].
        logits[0]:fp_logits
        logits[1]:fe_logits
    Returns:
      loss: Loss tensor of type float.
    """
    squre_error = tf.reduce_sum(tf.pow(logits[0] - landmarks_placeholder, 2)) / (LANDMARKS_LENGTH)
    squre_error_mean = tf.reduce_mean(squre_error, name='squre_error_mean')

    labels = tf.to_int64(labels_placeholder)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits[1]))
    xentropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', squre_error_mean*0.5 + xentropy_mean*0.5)
    tf.summary.scalar('squre_error_mean', squre_error_mean)
    tf.summary.scalar('xentropy_mean', xentropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def training(total_loss, init_learning_rate, global_step):
    lr = tf.train.exponential_decay(init_learning_rate,
                                    global_step,
                                    1000,
                                    0.96,
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