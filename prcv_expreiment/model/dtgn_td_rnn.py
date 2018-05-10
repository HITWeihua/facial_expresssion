import tensorflow as tf


IMAGE_SIZE = 64
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

CK_NUM_CLASSES = 8
CK_SIMPLE_NUM = 6
CK_LANDMARKS_LENGTH = 68*2*CK_SIMPLE_NUM

OULU_NUM_CLASSES = 6
OULU_SIMPLE_NUM = 7
OULU_LANDMARKS_LENGTH = 68*2*OULU_SIMPLE_NUM

MMI_NUM_CLASSES = 6
MMI_SIMPLE_NUM = 15
MMI_LANDMARKS_LENGTH = 68*2*MMI_SIMPLE_NUM

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


def inference(landmarks, keep_prob, is_train, batch_size_placeholder):
    with tf.variable_scope('block_top'):
        for i in range(OULU_SIMPLE_NUM):
            if i == 0:
                pass
                # inner_features_concat = landmarks[:, i, :]
            elif i == 1:
                inner_features_concat = landmarks[:, i, :] - landmarks[:, i-1, :]
                # inner_features_concat = tf.reshape(inner_features_concat, [-1, 1, 136])
            else:
                inner_features_concat = tf.concat([inner_features_concat, landmarks[:, i, :] - landmarks[:, i-1, :]], axis=-1)
    inner_features_concat = tf.reshape(inner_features_concat, [-1, 6, 136])
    n_hiddens = 32  # 隐层节点数
    n_layers = 2  # LSTM layer 层数

    # lstm_cell = rnn.BasicLSTMCell(num_units=n_hiddens, forget_bias=1.0, state_is_tuple=True)
    # lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    # mlstm_cell = rnn.MultiRNNCell([lstm_cell] * 2, state_is_tuple=True)
    # init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

    def attn_cell():
        lstm_cell = tf.contrib.rnn.BasicRNNCell(n_hiddens)

        with tf.name_scope('lstm_dropout'):
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

    enc_cells = []
    enc_cells2 = []
    for i in range(0, n_layers):
        enc_cells.append(attn_cell())
        enc_cells2.append(attn_cell())
    with tf.name_scope('lstm_cells_layers'):
        mlstm_cell = tf.contrib.rnn.MultiRNNCell(enc_cells, state_is_tuple=True)
        mlstm_cell2 = tf.contrib.rnn.MultiRNNCell(enc_cells2, state_is_tuple=True)
    # 全零初始化 state
    _init_state = mlstm_cell.zero_state(batch_size_placeholder, dtype=tf.float32)
    _init_state_bw = mlstm_cell2.zero_state(batch_size_placeholder, dtype=tf.float32)
    # dynamic_rnn 运行网络
    outputs, states = tf.nn.bidirectional_dynamic_rnn(mlstm_cell, mlstm_cell2, inner_features_concat, initial_state_fw=_init_state,
                                                      initial_state_bw=_init_state_bw, dtype=tf.float32,
                                                      time_major=False)
    # 输出
    outputs = tf.concat(outputs, 2)
    outputs = tf.transpose(outputs, [1, 0, 2])
    weights = weight_variable([2 * n_hiddens, OULU_NUM_CLASSES], stddev=0.1, name='weights', wd=0.01)
    biases = bias_variable([OULU_NUM_CLASSES], name='biases')
    # return tf.matmul(outputs[:,-1,:], Weights) + biases
    return tf.matmul(outputs[-1], weights) + biases


    # with tf.variable_scope('dtgn_fc1'):
    #     weights = weight_variable([OULU_LANDMARKS_LENGTH, 100], stddev=0.1, name='weights', wd=0.01)
    #     biases = bias_variable([100], name='biases')
    #     fc_1 = tf.nn.relu(tf.matmul(landmarks, weights) + biases)
    #     variable_summaries(fc_1, 'fc1')
    #     # fc_1_drop = tf.nn.dropout(fc_1, keep_prob)
    #
    # # fc2
    # with tf.variable_scope('dtgn_fc2'):
    #     weights = weight_variable([100, 600], stddev=0.1, name='weights', wd=0.01)
    #     biases = bias_variable([600], name='biases')
    #     fc_2 = tf.nn.relu(tf.matmul(fc_1, weights) + biases)
    #     variable_summaries(fc_2, 'fc2')
    #     fc2_drop = tf.nn.dropout(fc_2, keep_prob)
    #
    # # fc3 facial expression
    # with tf.variable_scope('dtgn_fc3_ep'):
    #     weights = weight_variable([600, OULU_NUM_CLASSES], stddev=0.1, name='weights', wd=0.01)
    #     biases = bias_variable([OULU_NUM_CLASSES], name='biases')
    #     fe_logits = tf.matmul(fc2_drop, weights) + biases
    #
    # return fe_logits


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
                                    20000,
                                    0.1,  # 0.96  0.3
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