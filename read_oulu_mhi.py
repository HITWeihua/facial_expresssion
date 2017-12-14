import tensorflow as tf
NUM_CLASSES = 6

def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename], num_epochs=10000)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   # 返回文件名和文件
    print('read train data.')
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([NUM_CLASSES], tf.float32),
                                           'images_mhi': tf.FixedLenFeature([64*64*8], tf.float32),
                                       })
    img = tf.cast(features['images_mhi'], tf.float32)
    images1 = tf.slice(img, [0], [64 * 64 * 7])
    images1 = tf.reshape(images1, [64, 64, 7])

    images2 = tf.slice(img, [64*64*7], [64*64*1])
    images2 = tf.reshape(images2, [64, 64, 1])
    # images = tf.concat([images1, images2], 2)
    label = tf.cast(features['label'], tf.float32)
    return images1, label


def read_and_decode_4_test(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename], num_epochs=10)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   # 返回文件名和文件
    print('read test data.')
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([NUM_CLASSES], tf.float32),
                                           'images_mhi': tf.FixedLenFeature([64*64*8], tf.float32),  # 24576+816=25392
                                       })
    img = tf.cast(features['images_mhi'], tf.float32)
    images1 = tf.slice(img, [0], [64 * 64 * 7])
    images1 = tf.reshape(images1, [64, 64, 7])

    images2 = tf.slice(img, [64 * 64 * 7], [64 * 64 * 1])
    images2 = tf.reshape(images2, [64, 64, 1])
    # images = tf.concat([images1, images2], 2)
    label = tf.cast(features['label'], tf.float32)
    return images1, label