import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

from model import resnet_dtan_v3
from model import dtan

GPU_NUM = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUM
SIMPLE_NUM = 7
LANDMARK_LENGTH = 68*2*SIMPLE_NUM
NUM_CLASSES = 6


def placeholder_inputs():
    images_placeholder = tf.placeholder(tf.float32, shape=[None, 64, 64, SIMPLE_NUM])
    # landmarks_placeholder = tf.placeholder(tf.float32, shape=[None, LANDMARK_LENGTH])
    labels_placeholder = tf.placeholder(tf.int32, shape=(None, NUM_CLASSES))
    keep_prob = tf.placeholder("float")
    is_train = tf.placeholder(tf.bool, name='phase_train')
    return images_placeholder, labels_placeholder, keep_prob, is_train


def fill_feed_dict(img, l, keep, is_train_value, images_placeholder, labels_placeholder, keep_prob, is_train):

    images_feed = img
    label_feed = l
    feed_dict = {
        images_placeholder: images_feed,
        labels_placeholder: label_feed,
        keep_prob: keep,
        is_train: is_train_value
    }
    return feed_dict


def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename], num_epochs=10000)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   # 返回文件名和文件
    print('read train data.')
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([NUM_CLASSES], tf.float32),
                                           'img_landmarks_raw': tf.FixedLenFeature([29624], tf.float32),
                                       })
    img = tf.cast(features['img_landmarks_raw'], tf.float32)
    images = tf.slice(img, [0], [dtan.IMAGE_PIXELS*dtan.SIMPLE_NUM])
    images = tf.reshape(images, [dtan.IMAGE_SIZE, dtan.IMAGE_SIZE, SIMPLE_NUM])
    # landmark = tf.slice(img, [dtan.IMAGE_PIXELS*dtan.SIMPLE_NUM], [LANDMARK_LENGTH])
    label = tf.cast(features['label'], tf.float32)
    return images, label


def read_and_decode_4_test(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename], num_epochs=10)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   # 返回文件名和文件
    print('read test data.')
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([NUM_CLASSES], tf.float32),
                                           'img_landmarks_raw': tf.FixedLenFeature([29624], tf.float32),  # 24576+816=29624
                                       })
    img = tf.cast(features['img_landmarks_raw'], tf.float32)
    images = tf.slice(img, [0], [dtan.IMAGE_PIXELS*dtan.SIMPLE_NUM])
    images = tf.reshape(images, [dtan.IMAGE_SIZE, dtan.IMAGE_SIZE, SIMPLE_NUM])
    # landmark = tf.slice(img, [dtan.IMAGE_PIXELS*dtan.SIMPLE_NUM], [LANDMARK_LENGTH])
    label = tf.cast(features['label'], tf.float32)
    return images, label


def run_training(fold_num, train_tfrecord_path, test_tfrecord_path, train_batch_size=60, test_batch_size=30):
    with tf.Graph().as_default():
        # with tf.device('/gpu:'+GPU_NUM):
        images, label = read_and_decode(train_tfrecord_path)
        # 使用shuffle_batch可以随机打乱输入
        images_batch, label_batch = tf.train.shuffle_batch([images, label], batch_size=train_batch_size, capacity=2000,
                                                        min_after_dequeue=1000)

        images_test, label_test = read_and_decode_4_test(test_tfrecord_path)
        # 使用shuffle_batch可以随机打乱输入
        images_batch_test, label_batch_test = tf.train.batch([images_test, label_test], batch_size=test_batch_size, capacity=2000)

        # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder, keep_prob, is_train = placeholder_inputs()

        # Build a Graph that computes predictions from the inference model.
        fe_logits = resnet_dtan_v3.inference(images_placeholder, keep_prob, is_train)

        # Add to the Graph the Ops for loss calculation.
        loss = dtan.loss(fe_logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        global_step = tf.Variable(0, trainable=False)
        train_op = dtan.training(loss, flags.learning_rate, global_step)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = dtan.evaluation(fe_logits, labels_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Create a saver for writing training checkpoints.
        # saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, gpu_options= gpu_options)) as sess:

            # Instantiate a SummaryWriter to output summaries and the Graph.
            train_writer = tf.summary.FileWriter('./summaries/summaries_graph_1217(3)/'+str(fold_num)+'/train', sess.graph)
            test_writer = tf.summary.FileWriter('./summaries/summaries_graph_1217(3)/'+str(fold_num)+'/test', sess.graph)

            # And then after everything is built:

            # Run the Op to initialize the variables.
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            img_test, l_test = sess.run([images_batch_test, label_batch_test])
            test_feed_dict = fill_feed_dict(img_test, l_test, 1.0, False, images_placeholder, labels_placeholder, keep_prob, is_train)
            # Start the training loop.
            last_train_correct = []
            last_test_correct = []
            for step in range(flags.max_steps):
                start_time = time.time()

                # Fill a feed dictionary with the actual set of images and labels
                # for this particular training step.
                img, l = sess.run([images_batch, label_batch])
                feed_dict = fill_feed_dict(img, l, 0.5, True, images_placeholder, labels_placeholder, keep_prob, is_train)

                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.  To
                # inspect the values of your Ops or variables, you may include them
                # in the list passed to sess.run() and the value tensors will be
                # returned in the tuple from the call.
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

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
                    if step > flags.max_steps - 10*50:
                        last_train_correct.append(train_correct)
                        last_test_correct.append(test_correct)
                    if (step + 1) == flags.max_steps:
                        print(last_train_correct)
                        print(last_test_correct)
                        print(np.array(last_train_correct).mean())
                        print(np.array(last_test_correct).mean())
            coord.request_stop()
            coord.join(threads)
    return last_train_correct, last_test_correct


def main(_):
    base_path = "./oulu_el_joint"
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
        train, test = run_training(i, train_tfrecord_path, test_tfrecord_path, train_batch_size=64, test_batch_size=test_batch_size)
        train_correct.append(train)
        test_correct.append(test)
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
        default=0.001,
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
        default=3000,
        help='max steps initial 3000.'

    )
    flags, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
