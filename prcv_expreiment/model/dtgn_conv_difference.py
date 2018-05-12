import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

# from model import temporal_difference_v0 as td_model
# from model import temporal_difference_sw as td_model
sys.path.append(os.path.abspath('.'))
print(os.path.abspath('.'))

GPU_NUM = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUM

IMAGE_SIZE = 64
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

CK_NUM_CLASSES = 8
CK_SIMPLE_NUM = 6
CK_LANDMARKS_LENGTH = 68 * 2 * CK_SIMPLE_NUM

OULU_NUM_CLASSES = 6
OULU_SIMPLE_NUM = 7
OULU_LANDMARKS_LENGTH = 68 * 2 * OULU_SIMPLE_NUM

MMI_NUM_CLASSES = 6
MMI_SIMPLE_NUM = 15
MMI_LANDMARKS_LENGTH = 68 * 2 * MMI_SIMPLE_NUM

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
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
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


def block_top(image, is_train):
    image = tf.reshape(image, [-1, 68, 2, 1])
    kernel1 = weight_variable([1, 3, 1, 16], stddev=0.1, name='weights', wd=0.0)
    biases1 = bias_variable([16], name='biases')
    conv1 = conv2d(image, kernel1) + biases1
    conv1_bn = batch_norm(conv1, 16, is_train)
    conv1_activation = ACTIVATION(conv1_bn, name='activate')
    return conv1_activation


def inference(images, keep_prob, is_train):
    images = tf.reshape(images, [-1, 7, 68, 2])
    with tf.variable_scope('block_top'):
        frames_features = block_top(images[:, 0, :, :], is_train)
        frames_features1 = block_top(images[:, 3, :, :], is_train)
        frames_features2 = block_top(images[:, 6, :, :], is_train)
        inner_feature_concat0 = frames_features1 - frames_features
        inner_feature_concat = tf.concat([inner_feature_concat0, frames_features2 - frames_features1], axis=-1)

    with tf.variable_scope('fc1'):
        weights = weight_variable([32 * 2 * 68, 600], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([600], name='biases')
        h_pool = tf.reshape(inner_feature_concat, [-1, 32 * 2 * 68])
        fc1 = tf.nn.relu(tf.matmul(h_pool, weights) + biases)
        # variable_summaries(fc1)
        fc1_drop = tf.nn.dropout(fc1, keep_prob)
    # fc2  facial point
    # with tf.variable_scope('fc2'):
    #    weights = weight_variable([4096, LANDMARKS_LENGTH], stddev=0.1, name='weights', wd=0.01)
    #    biases = bias_variable([LANDMARKS_LENGTH], name='biases')
    #    fp_logits = tf.matmul(fc_all_drop, weights) + biases

    # fc3 facial expression
    with tf.variable_scope('fc3'):
        weights = weight_variable([600, OULU_NUM_CLASSES], stddev=0.1, name='weights', wd=0.01)
        biases = bias_variable([OULU_NUM_CLASSES], name='biases')
        fe_logits = tf.matmul(fc1_drop, weights) + biases

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
                                    10000,
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


def placeholder_inputs():
    # images_placeholders = []
    # for i in range(SIMPLE_NUM):
    #     images_placeholders.append(tf.placeholder(tf.float32, shape=[None, 64, 64, 1]))
    # images_placeholder = tf.placeholder(tf.float32, shape=[None, 64, 64, model.OULU_SIMPLE_NUM])
    landmarks_placeholder = tf.placeholder(tf.float32, shape=[None, OULU_LANDMARKS_LENGTH])
    labels_placeholder = tf.placeholder(tf.int32, shape=(None, OULU_NUM_CLASSES))
    keep_prob = tf.placeholder("float")
    is_train = tf.placeholder(tf.bool, name='phase_train')
    return landmarks_placeholder, labels_placeholder, keep_prob, is_train


def fill_feed_dict(ld, l, keep, is_train_value, landmarks_placeholder, labels_placeholder, keep_prob, is_train):
    images_feed = ld
    label_feed = l
    feed_dict = {
        landmarks_placeholder: images_feed,
        labels_placeholder: label_feed,
        keep_prob: keep,
        is_train: is_train_value
    }
    return feed_dict


def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename], num_epochs=10000)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    print('read train data.')
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([OULU_NUM_CLASSES], tf.float32),
                                           'img_landmarks_raw': tf.FixedLenFeature([29624], tf.float32),
                                       })
    img = tf.cast(features['img_landmarks_raw'], tf.float32)
    # images = tf.slice(img, [0], [model.IMAGE_PIXELS*model.CK_SIMPLE_NUM])
    # images = tf.reshape(images, [model.IMAGE_SIZE, model.IMAGE_SIZE, model.CK_SIMPLE_NUM])
    landmark = tf.slice(img, [IMAGE_PIXELS * OULU_SIMPLE_NUM], [OULU_LANDMARKS_LENGTH])
    label = tf.cast(features['label'], tf.float32)
    return landmark, label


def read_and_decode_4_test(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename], num_epochs=10)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    print('read test data.')
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([OULU_NUM_CLASSES], tf.float32),
                                           'img_landmarks_raw': tf.FixedLenFeature([29624], tf.float32),
                                       # 24576+816=29624
                                       })
    img = tf.cast(features['img_landmarks_raw'], tf.float32)
    # images = tf.slice(img, [0], [model.IMAGE_PIXELS*model.CK_SIMPLE_NUM])
    # images = tf.reshape(images, [model.IMAGE_SIZE, model.IMAGE_SIZE, model.CK_SIMPLE_NUM])
    landmark = tf.slice(img, [IMAGE_PIXELS * OULU_SIMPLE_NUM], [OULU_LANDMARKS_LENGTH])
    label = tf.cast(features['label'], tf.float32)
    return landmark, label


def run_training(fold_num, train_tfrecord_path, test_tfrecord_path, train_batch_size=60, test_batch_size=30):
    with tf.Graph().as_default():
        # with tf.device('/gpu:'+GPU_NUM):
        images, label = read_and_decode(train_tfrecord_path)
        # 使用shuffle_batch可以随机打乱输入
        images_batch, label_batch = tf.train.shuffle_batch([images, label], batch_size=train_batch_size, capacity=1000,
                                                           min_after_dequeue=800)

        images_test, label_test = read_and_decode_4_test(test_tfrecord_path)
        # 使用shuffle_batch可以随机打乱输入
        images_batch_test, label_batch_test = tf.train.batch([images_test, label_test], batch_size=test_batch_size,
                                                             capacity=1000)

        # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder, keep_prob, is_train = placeholder_inputs()

        # Build a Graph that computes predictions from the inference model.
        fe_logits = inference(images_placeholder, keep_prob, is_train)

        # Add to the Graph the Ops for loss calculation.
        loss_1 = loss(fe_logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        global_step = tf.Variable(0, trainable=False)
        train_op = training(loss_1, flags.learning_rate, global_step)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = evaluation(fe_logits, labels_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Create a saver for writing training checkpoints.
        # saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                              gpu_options=gpu_options)) as sess:

            # Instantiate a SummaryWriter to output summaries and the Graph.
            train_writer = tf.summary.FileWriter('./summaries_new1/summaries_graph_0420/' + str(fold_num) + '/train',
                                                 sess.graph)
            test_writer = tf.summary.FileWriter('./summaries_new1/summaries_graph_0420/' + str(fold_num) + '/test',
                                                sess.graph)

            # And then after everything is built:

            # Run the Op to initialize the variables.
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            img_test, l_test = sess.run([images_batch_test, label_batch_test])
            test_feed_dict = fill_feed_dict(img_test, l_test, 1.0, False, images_placeholder, labels_placeholder,
                                            keep_prob, is_train)
            # Start the training loop.
            last_train_correct = []
            last_test_correct = []
            for step in range(flags.max_steps):
                start_time = time.time()

                # Fill a feed dictionary with the actual set of images and labels
                # for this particular training step.
                img, l = sess.run([images_batch, label_batch])
                feed_dict = fill_feed_dict(img, l, 0.9, True, images_placeholder, labels_placeholder, keep_prob,
                                           is_train)

                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.  To
                # inspect the values of your Ops or variables, you may include them
                # in the list passed to sess.run() and the value tensors will be
                # returned in the tuple from the call.
                _, loss_value = sess.run([train_op, loss_1], feed_dict=feed_dict)

                duration = time.time() - start_time

                # Write the summaries and print an overview fairly often.
                if step % 100 == 0 or (step + 1) == flags.max_steps:
                    print('fold_num:{}'.format(fold_num))
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                    # Update the events file.
                    train_summary_str = sess.run(summary, feed_dict=feed_dict)
                    test_summary_str = sess.run(summary, feed_dict=test_feed_dict)
                    train_writer.add_summary(train_summary_str, step)
                    test_writer.add_summary(test_summary_str, step)
                    # summary_writer.flush()

                    print('Training Data Eval:')
                    train_correct = sess.run(eval_correct, feed_dict=feed_dict)
                    print('train_correct:{}'.format(train_correct))
                    print('Test Data Eval:')
                    test_correct = sess.run(eval_correct, feed_dict=test_feed_dict)
                    print('test_correct:{}\n\n'.format(test_correct))
                    # if (step + 1) == flags.max_steps:
                    if step > flags.max_steps - 10 * 50:
                        last_train_correct.append(train_correct)
                        last_test_correct.append(test_correct)
                    if (step + 1) == flags.max_steps:
                        # fe_logits_last_values = sess.run(fe_logits, feed_dict=test_feed_dict)
                        # np.savetxt('./summaries/summaries_graph_1219/' + str(fold_num) + '/logit.txt',
                        #            fe_logits_last_values)
                        # np.savetxt('./summaries/summaries_graph_1219/' + str(fold_num) + '/test_l.txt',
                        #            l_test)
                        print(last_train_correct)
                        print(last_test_correct)
                        print(np.array(last_train_correct).mean())
                        print(np.array(last_test_correct).mean())
            # saver_path = saver.save(sess, "/home/duheran/facial_expresssion/save/dtgn.ckpt")  # 将模型保存到save/model.ckpt文件
            # print("Model saved in file:", saver_path)
            coord.request_stop()
            coord.join(threads)

    return last_train_correct, last_test_correct


def main(_):
    base_path = "/home/duheran/facial_expresssion/oulu_el_joint"  # 2018.5.6
    train_correct = []
    test_correct = []
    for i in range(10):
        test_train_dir = os.path.join(base_path, str(i))
        test_train_files = os.listdir(test_train_dir)
        for file_name in test_train_files:
            if 'test' in file_name:
                test_file = file_name
            elif 'train' in file_name:
                train_file = file_name
        train_tfrecord_path = os.path.join(test_train_dir, train_file)
        test_tfrecord_path = os.path.join(test_train_dir, test_file)
        # test_batch_size = int(os.path.splitext(test_tfrecord_path)[0][-2:])
        test_batch_size = 48
        train, test = run_training(i, train_tfrecord_path, test_tfrecord_path, train_batch_size=64,
                                   test_batch_size=test_batch_size)
        train_correct.append(train)
        test_correct.append(test)
    print('Diffconv OULU result')
    print(np.array(train_correct).shape)
    print(np.array(test_correct).shape)
    print(np.array(train_correct).mean(axis=1))
    print(np.array(test_correct).mean(axis=1))
    print("train_correct:{}".format(np.array(train_correct).mean()))
    print("test_correct:{}".format(np.array(test_correct).mean()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,  # 0.001
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=64,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=35000,
        help='max steps initial 3000.'

    )
    flags, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
