import argparse
import ntpath
import glob
import os

import tensorflow as tf
import numpy as np
import youtube_dl
import cv2


def _int64_feature(value):
    """Returns an int64_list."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Returns an bytes_list."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def diveision_zero(x, y):
    '''
    x / y
    Helper function to divide number by zero.
    IF divided by zero zero will be return.
    '''

    try:
        return x / y
    except ZeroDivisionError:
        return 0

def frame_encoder(frame):
    '''
    :param frame:

    This Function will encode frame to proto buffer.
    '''

    # Encoding Key.
    feature = {
        'frame': _bytes_feature(frame)
    }

    # Encode
    proto = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize proto buffer to string.
    proto = proto.SerializeToString()

    return proto

def frame_decoder(proto):
    '''
    :param proto: Proto buffer to be decoded.
    :return: tensor with decode frame.

    This Function will decode frame from proto buffer.
    '''

    # Decoding Key.
    features = {
        'frame': tf.FixedLenFeature([], tf.string)
    }

    # Decode.
    sample = tf.parse_single_example(proto, features)
    frame = tf.image.decode_image(sample['frame'])

    return frame


def worker(work: list,
           save_dir: str,
           target_fps: int = 0,
           target_resolution: tuple = None,
           download: bool = True,
           use_subtitles: bool = False,
           download_buffer_dir: str = '',
           youtube_base: str = 'https://www.youtube.com/watch?v='):
    '''
    :param work: List with path to existing videos (if so download need to be True (default))
    or list with youtube video IDs (if so download need to be False).
    :param save_dir: Directory where the finished TF.record's are saved.
    :param target_fps: The fps from the TF.record, if 0 original download fps will be kept (default).
    :param target_resolution: Tuple with (width, height) resolution.
    If None original download resolution will be kept (default).
    :param download: Bool if it needs to download the video or just proses it. If download=True (default) youtube ID's
    needs to be given and if download=False path to videos needs to be given.
    :param use_subtitles: Bool, if true Text will be used to. (not implemented jet)
    :param download_buffer_dir: Directory where YoutubDL will download the videos (only if download is True (default)).
    It is recommended to use a RAM Disk as buffer directory.
    :param youtube_base: Youtube base string https://www.youtube.com/watch?v=.

    This function will download youtube videos and proses them and save than as TF.record files.
    '''

    # Check if video needs to be downloaded.
    if download:
        # Configer base paramiter for YoutubeDL.
        ydl_opts = {'outtmpl': download_buffer_dir + '%(id)s.%(ext)s'}

        # Check if subtitles are required, if so add parameter to YoutubeDL config.
        if use_subtitles:
            ydl_opts['writesubtitles'] = str(use_subtitles)
            ydl_opts['writeautomaticsub'] = str(use_subtitles)
            ydl_opts['subtitlesformat'] = 'vtt'
            ydl_opts['subtitleslangs'] = ['en']

        # Creat Youtube Downloader.
        youtube_downloader = youtube_dl.YoutubeDL(ydl_opts)

    # Loop to list of work.
    for wor in work:

        # Check if video needs to be downloaded.
        if download:

            # Download video.
            youtube_downloader.download([youtube_base + wor])

            # Change ID string to video path, this will be needed so we can creat TF.record from the download.
            wor = os.path.join(download_buffer_dir, (wor + '.*'))
            wor = glob.glob(wor)[0]
            wor = os.path.join(download_buffer_dir, wor)

        # Setup CV2 video reader.
        video_cap = cv2.VideoCapture(wor)
        success, frame = video_cap.read()
        frame_count = 0

        # Get base video FPS, this is needed to align language and video.
        fps_raw = video_cap.get(cv2.CAP_PROP_FPS)

        # Calculate Modulo number.
        fps_split = diveision_zero(round(fps_raw), target_fps)

        # Create TF Record Writer
        _save_name = save_dir + ntpath.basename(wor) + '.tfrecord'
        with tf.io.TFRecordWriter(_save_name) as tf_writer:
            while success:

                # Check if frame needs to be saved.
                if frame_count % fps_split == 0:

                    # Do resizing if required.
                    if target_resolution is not None:
                        frame = cv2.resize(frame, target_resolution)

                    # convert BGR to RGB frames.
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Encode frame to image string.
                    frame = cv2.imencode('.jpg', frame)[1].tostring()

                    # Encode frame to proto puffer.
                    proto = frame_encoder(frame)

                    # Write proto buffer to TF.record file.
                    tf_writer.write(proto)

                # Get next frame.
                success, frame = video_cap.read()
                frame_count += 1

        # Release video capture.
        video_cap.release()

        # Remove video file if it was downloaded.
        if download:
            os.remove(wor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    worker(['CyrgzzxQPzk', '4MkrEMjPk24', 'JAZdUmsCvsw'], '', target_fps=1, target_resolution=(320, 176))

    '''
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

    '''
