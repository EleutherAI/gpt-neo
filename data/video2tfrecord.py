import tensorflow as tf
import numpy as np
import cv2
import os


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def frame_encoder(frame):

    feature = {
        'frame': _bytes_feature(frame)
    }

    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto

def frame_decoder(tfrecord):
    features = {
        'frame': tf.FixedLenFeature([], tf.string)
    }

    sample = tf.parse_single_example(tfrecord, features)
    frame = tf.image.decode_image(sample['frame'])

    return frame





if __name__ == '__main__':



    video_cap = cv2.VideoCapture('/opt/project/7vPNcnYWQ4.mp4')
    fps = round(video_cap.get(cv2.CAP_PROP_FPS))
    success, frame = video_cap.read()
    frame_count = 0
    print('fps', fps)

    with tf.io.TFRecordWriter('/opt/project/test.tfrecord') as tf_writer:

        while success:
            if frame_count % fps == 0 or frame_count == 0:
                frame = cv2.resize(frame, (426, 240))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.imencode('.jpg', frame)[1].tostring()
                proto = frame_encoder(frame)
                tf_writer.write(proto.SerializeToString())

            success, frame = video_cap.read()
            frame_count += 1
            
    video_cap.release()



    dataset = tf.data.TFRecordDataset(['/opt/project/test.tfrecord'])
    dataset = dataset.map(frame_decoder)
    iterator = dataset.make_one_shot_iterator()
    next_frame_data = iterator.get_next()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        writer = cv2.VideoWriter('/opt/project/test.avi', cv2.VideoWriter_fourcc(*"MJPG"), 1, (426, 240))

        try:
            # Keep extracting data till TFRecord is exhausted
            while True:
                frame_data = sess.run(next_frame_data)
                writer.write(np.array(frame_data).astype('uint8'))
        except:
            writer.release()


