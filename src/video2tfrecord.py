import multiprocessing
import argparse
import datetime
import ntpath
import glob
import json
import re
import os

from transformers import GPT2Tokenizer
from google.cloud import storage
import tensorflow as tf
import numpy as np
import youtube_dl
import cv2


def _int64_feature(value):
    """Returns an int64_list."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


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


def frame_encoder(frame, text_tokens=None, skip_frame: list = [False]):
    '''
    :param frame: A byte String containing a jpg encoded image.
    :param text_tokens: A list containing int ped tokens.
    :param skip_frame: A list containing a single bool that
    determines if this frame include an image or just text.

    This Function will encode frame to proto buffer.
    '''

    # Encoding Key.
    feature = {
        'frame': _bytes_feature(frame)
    }

    if text_tokens is not None:
        feature.update({'tokens': _int64_feature(text_tokens),
                        'skip_frame': _int64_feature(skip_frame)})

    # Encode
    proto = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize proto buffer to string.
    proto = proto.SerializeToString()

    return proto


def get_decoder(language_token_num_per_frame=0):
    '''
    :param language_token_num_per_frame: The number of language tokens per single frame.
    If this is 0 (default) language tokens are disabled.

    This function will return a frame decoder function, that can than be used to decode tf.records.
    '''

    decode_language_token = language_token_num_per_frame > 0

    # Decoding Key.
    features = {
        'frame': tf.FixedLenFeature([], tf.string)
    }

    if decode_language_token:
        features.update({'tokens': tf.FixedLenFeature([language_token_num_per_frame], tf.int64),
                         'skip_frame': tf.FixedLenFeature([], tf.int64)})

    def frame_decoder(proto):
        '''
        :param proto: Proto buffer to be decoded.
        :return: tensor with decode frame.

        This Function will decode frame from proto buffer.
        '''

        # Decode.
        sample = tf.parse_single_example(proto, features)

        if decode_language_token:
            frame = (tf.image.decode_image(sample['frame']), sample['tokens'], sample['skip_frame'])

        else:
            frame = tf.image.decode_image(sample['frame'])

        return frame

    return frame_decoder


def split_equal(id: list, num: int):

    id_split = [[] for i in range(num)]

    for i in id:
        l = [len(s) for s in id_split]
        pos = np.argmin(l)

        id_split[pos].append(i)

    return id_split


def decode_vtt(content: str):
    '''
    :param content: String with the of the .vtt file.
    :return: String with combined text, List with Strings split at time stamps and List with float second time stamps.

    This Function decodes a vtt to get the contend with  time stamps.
    '''

    # Split the content at line brake and check if word level time stamps are in the line.
    content = [l for l in content.split('\n') if '<c>' in l]

    # Connect list of strings back together.
    content = "".join(content)

    # Split String at time stamp headers.
    content = content.split('><c>')

    # Create output lists.
    words = []
    stamp = []

    # Loop word and time stamp string list.
    for c in content:

        # Split word and time stamp part.
        word = c[:-12]
        stam = c[-12:]

        # Clean word string.
        if not '</c><' in word:
            word = word.replace('</c>', ' ')

        word = word.replace('</c>', '')
        word = word.replace('<', '')
        word = word.lstrip().rstrip()

        # Check if time stamp is in stamp string.
        if ':' in stam and '.' in stam:

            # Split time stamp string at time punctuation marks.
            stam = stam.split(':')
            stam = stam[:-1] + stam[-1].split('.')

            # Converting time stamp string in to second based float.
            stam = datetime.timedelta(hours=int(stam[0]), minutes=int(stam[1]), seconds=int(stam[2]), milliseconds=int(stam[3]))
            stam = stam.total_seconds()

            # add word string and second based float to output list.
            words.append(word)
            stamp.append(stam)
        else:
            # If no time stamp contain in time stamp part we assume that it is a another word.
            # If it as a another word we add it to the previous word string.
            if len(words) > 0:
                words[-1] = words[-1] + " " + c.replace('</c>', '').replace('<', '').lstrip().rstrip()

    return ' '.join(words), words, stamp


def bpe_with_word_split(enc: GPT2Tokenizer, words: list, text: str):
    '''
    :param enc: BPE encoder to be used.
    :param words: List with word strings split at time stamps.
    :param text: String with total text. the sum of text in words list should be the same as the content of the text string.
    :return: List with list containing BPE tokens.

    This function will encode a text using a given tokenizer and
    that split it back up according to the string split provided by word list.
    '''

    # Encode text with tokenizer.
    tokens = enc.encode(text, max_length=None)

    # Decode single token.
    pair_split_word = [enc.decode(token).replace(' ', '') for token in tokens]

    # Creat output list.
    bpe_list = []

    # Creat token split index.
    idx = 0

    # Loop through word list.
    for word in words:

        # Create buffer list to holt the tokens for one time stamp interval.
        buffer = []

        # Buffer word string and remove empty space between single words.
        word_buffer = word.replace(' ', '')

        # Continue looping es long  idx is lower as token count and
        # the current token string is in string from time stamp interval.
        while (idx < len(pair_split_word)) and (pair_split_word[idx] in word):

            # Get the length of the token string.
            token_word_len = len(pair_split_word[idx])

            # Check if the token string is at the end of the time stamp interval string.
            # This is important for tokens like 'a' and 'i'.
            if word_buffer[:token_word_len] == pair_split_word[idx]:

                # Add token to buffer and remove the string length of the token from the time stamp interval string.
                buffer.append(tokens[idx])
                word_buffer = word_buffer[token_word_len:]
                idx += 1
            else:
                break

        # Append buffer list to output list.
        bpe_list.append(buffer)

    return bpe_list


def worker(work: list,
           save_dir,
           target_fps: int = 0,
           target_resolution: tuple = None,
           download: bool = True,
           keep_buffer_download: bool = False,
           download_buffer_dir: str = '',
           use_subtitles: bool = False,
           enc: GPT2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2'),
           language_tokens_per_frame: int = 4,
           skip_if_no_subtitles: bool = True,
           youtube_base: str = 'https://www.youtube.com/watch?v='):
    '''
    :param work: List with path to existing videos (if so download need to be True (default))
    or list with youtube video IDs (if so download need to be False).
    :param save_dir: Directory where the finished TF.record's are saved or google blob to upload TF.record.
    :param target_fps: The fps from the TF.record, if 0 original download fps will be kept (default).
    :param target_resolution: Tuple with (width, height) resolution.
    If None original download resolution will be kept (default).
    :param download: Bool if it needs to download the video or just proses it. If download=True (default) youtube ID's
    needs to be given and if download=False path to videos needs to be given.
    :param keep_buffer_download: If True the downloaded files will not be deleted after the tfrecord got created.
    :param use_subtitles: Bool, if true Text will be used to.
    :param enc: The BPE token encoder.
    :param language_tokens_per_frame: The number of language tokens encoded per single frame.
    :param skip_if_no_subtitles: If True the video will be skipped if no subtitles are available.
    (only if use_subtitles is True)
    :param download_buffer_dir: Directory where YoutubDL will download the videos (only if download is True (default)).
    It is recommended to use a RAM Disk as buffer directory.
    :param youtube_base: Youtube base string https://www.youtube.com/watch?v=.

    This function will download youtube videos and proses them and save than as TF.record files.
    '''

    # Check if TF.record are uploaded to google cloud storage.
    if type(save_dir) is storage.Blob:
        cloud_storage = True
        buffer_save_dir = download_buffer_dir
    else:
        cloud_storage = False
        buffer_save_dir = save_dir

    # Get the vocab size from the BPE encoder.
    vocab_size = len(enc.get_vocab())

    pading_frame = cv2.imencode('.jpg', np.array([[[255]]]))[1].tostring()

    # Check if video needs to be downloaded.
    if download:
        # Configer base paramiter for YoutubeDL.
        ydl_opts = {'outtmpl': download_buffer_dir + '%(id)s.%(ext)s', 'socket-timeout': 600}

        # Check if subtitles are required, if so add parameter to YoutubeDL config.
        if use_subtitles:
            ydl_opts['writeautomaticsub'] = str(use_subtitles)
            ydl_opts['subtitlesformat'] = 'vtt'
            #ydl_opts['subtitleslangs'] = ['en']

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
            wor = [w for w in glob.glob(wor) if '.vtt' not in w][0]
            wor = os.path.join(download_buffer_dir, wor)

        # Assume by default the subtitles are available.
        subtitles_available = True

        if use_subtitles:
            vtt = os.path.splitext(wor)[0]
            vtt = os.path.join(download_buffer_dir, (vtt + '.*'))
            vtt = [v for v in glob.glob(vtt) if '.vtt' in v]

            if len(vtt) > 0:
                vtt = os.path.join(download_buffer_dir, vtt[0])

                with open(vtt, "r") as r:
                    text, words, stamp = decode_vtt(r.read())

                bpe_list = bpe_with_word_split(enc, words, text)

            else:
                subtitles_available = False

        # Check if this video need to be skipt. This will happen if subtitles are required but are not available.
        if not use_subtitles or \
                (subtitles_available and not skip_if_no_subtitles) or \
                (subtitles_available and skip_if_no_subtitles):

            # Setup CV2 video reader.
            video_cap = cv2.VideoCapture(wor)
            success, frame = video_cap.read()
            frame_count = 0

            # Get base video FPS, this is needed to align language and video.
            fps_raw = video_cap.get(cv2.CAP_PROP_FPS)

            # Calculate Modulo number.
            fps_split = diveision_zero(round(fps_raw), target_fps)

            # Create TF Record Writer
            _save_name = buffer_save_dir + os.path.splitext(ntpath.basename(wor))[0] + '.tfrecord'
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

                        if use_subtitles:
                            token_buffer = []
                            proto = []

                            while len(bpe_list) > 0 and stamp[0] < (frame_count + fps_split) * (1 / fps_raw):
                                token_buffer += bpe_list[0]
                                bpe_list.pop(0)
                                stamp.pop(0)


                            for i in range(0, len(token_buffer), language_tokens_per_frame):
                                buffer = token_buffer[i:i + language_tokens_per_frame]
                                buffer += [vocab_size + 1] * (language_tokens_per_frame - len(buffer))
                                skip_buffer = i > 0

                                proto.append(frame_encoder(pading_frame if skip_buffer else frame,
                                                           buffer,
                                                           [skip_buffer]))

                            if len(proto) <= 0:
                                proto.append(frame_encoder(frame,
                                                           [vocab_size + 1] * language_tokens_per_frame,
                                                           [False]))

                        else:
                            # Encode frame to proto buffer.
                            proto = [frame_encoder(frame)]

                        #print(frame_count, len(proto))

                        # Write proto buffer to TF.record file.
                        for p in proto:
                            tf_writer.write(p)

                    # Get next frame.
                    success, frame = video_cap.read()
                    frame_count += 1

            # close video file.
            video_cap.release()

        # Remove video file if it was downloaded.
        if download and not keep_buffer_download:
            os.remove(wor)
            if use_subtitles and subtitles_available:
                os.remove(vtt)

        if cloud_storage and not keep_buffer_download:
            save_dir.upload_from_filename(_save_name)
            os.remove(_save_name)



if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()

    enc = GPT2Tokenizer.from_pretrained('gpt2')

    f = open("7vPNcnYWQ4.en.vtt", "r")
    vtt = f.read()
    f.close()

    text, words, stamp = decode_vtt(vtt)
    bpe_list = encode_with_word_split(enc, words, text)
    '''

    '''
    duffer = []
    idx = 1
    while len(bpe_list) > 0:
        affer = []
        while len(bpe_list) > 0 and stamp[0] < idx:
            affer = affer + bpe_list[0]
            bpe_list.pop(0)
            stamp.pop(0)

        duffer.append(affer)
        idx += 1

    for i in range(len(duffer)):
        print(enc.decode(duffer[i]), duffer[i], i)
    '''

    '''
    content = os.listdir('/buffer/')
    content = ['/buffer/' + c for c in content if '.vtt' not in c]
    print(len(content), content)
    worker(content, '/video_only/', target_fps=1, target_resolution=(320, 176), keep_buffer_download=True,
           download_buffer_dir='/buffer/', use_subtitles=False, download=False)
    '''

    '''
    #id = json.load(open('channel_video_id_list.json'))

    content = os.listdir('/buffer/')
    content = ['/buffer/' + c for c in content if '.vtt' not in c]
    print(len(content), content)

    split_id = split_equal(content, 40)
    split_sum = 0

    for s in split_id:
        print(len(s), s)
        split_sum = split_sum + len(s)

    print('total split sum:', split_sum)

    work = []

    for c in split_id:
        p = multiprocessing.Process(target=worker, args=(c,
                                                         '/video_only/',
                                                         1,
                                                         (320, 176),
                                                         False,
                                                         False,
                                                         True,
                                                         '/buffer/',))

        p.start()
        work.append(p)

    for w in work:
        w.join()



    #worker(id, '/video_only/', target_fps=1, target_resolution=(320, 176), keep_buffer_download=True, download_buffer_dir='/buffer/', use_subtitles=True)
    '''

    '''
    video_cap = cv2.VideoCapture('/opt/project/7vPNcnYWQ4.mp4')
    fps = round(video_cap.get(cv2.CAP_PROP_FPS))
    success, frame = video_cap.read()
    frame_count = 0
    print('fps', fps)

    with tf.io.TFRecordWriter('/opt/project/test.tfrecord') as tf_writer:

        while success:
            if frame_count % fps == 0 or frame_count == 0:
                frame = cv2.resize(frame, 320, 176)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.imencode('.jpg', frame)[1].tostring()
                proto = frame_encoder(frame)
                tf_writer.write(proto.SerializeToString())

            success, frame = video_cap.read()
            frame_count += 1
            
    video_cap.release()
    '''


    worker(['/opt/project/7vPNcnYWQ4.mp4'],
           save_dir='/opt/project/',
           target_fps=1,
           target_resolution=(320, 176),
           download=False,
           use_subtitles=True)



    frame_decoder = get_decoder(language_token_len=4)

    dataset = tf.data.TFRecordDataset(['/opt/project/7vPNcnYWQ4.tfrecord'])
    dataset = dataset.map(frame_decoder)
    iterator = dataset.make_one_shot_iterator()
    next_frame_data = iterator.get_next()
    buffer_frame = None

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        writer = cv2.VideoWriter('/opt/project/test.avi', cv2.VideoWriter_fourcc(*"MJPG"), 1, (320, 176))

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 100)
        fontScale = 0.6
        fontColor = (255, 255, 255)
        lineType = 2

        try:
            # Keep extracting data till TFRecord is exhausted
            while True:
                frame_data, token, skip_frame = sess.run(next_frame_data)
                print(np.array(frame_data).shape, token, skip_frame)

                if not skip_frame:
                    frame = np.array(frame_data).astype('uint8')

                #frame = np.copy(buffer_frame)
                cv2.putText(frame, '[' + " ".join([str(t) for t in token]) + ']',
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)

                writer.write(frame)
        except:
            writer.release()


