import multiprocessing
import subprocess
import argparse
import datetime
import warnings
import ntpath
import random
import glob
import json
import os

from urllib.request import urlretrieve
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

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def downloader(url: str, filename: str, max_try: int=3):
    try_count = 0

    while try_count < max_try:
        try:
            urlretrieve(url, filename)
        except:
            try_count += 1
        else:
            return True

    print('Retry exceeded')

    if os.path.exists(filename):
        os.remove(filename)

    return False


def frame_encoder(frame, text_tokens=None, skip_frame: list = [False], mask: list = None):
    '''
    :param frame: A byte String containing a jpg encoded image.
    :param text_tokens: A list containing int ped tokens.
    :param skip_frame: A list containing a single bool that
    determines if this frame include an image or just text.
    :param mask: A int that determines when the padding tokens start.

    This Function will encode frame to proto buffer.
    '''

    # Encoding Key.
    feature = {
        'frame': _bytes_feature(frame)
    }

    if text_tokens is not None:
        feature.update({'tokens': _int64_feature(text_tokens),
                        'skip_frame': _int64_feature(skip_frame),
                        'mask': _int64_feature(mask)})

    # Encode
    proto = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize proto buffer to string.
    proto = proto.SerializeToString()

    return proto


def get_decoder(language_token_num_per_frame=0, frame_height=None, frame_width=None, color_channels=None):
    '''
    :param language_token_num_per_frame: The number of language tokens per single frame.
    If this is 0 (default) language tokens are disabled.
    :param frame_height:
    :param frame_width:
    :param color_channels:

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

        sample = tf.parse_single_example(proto, features)
        frame = tf.image.decode_image(sample['frame'])

        if decode_language_token:
            tokens = sample['tokens']
            skip_frame = sample['skip_frame']

            if skip_frame > 0:
                frame = tf.zeros(shape=(frame_height, frame_width, color_channels), dtype=tf.uint8)

            return frame, tokens, skip_frame

        return frame


    return tf.function(frame_decoder, experimental_compile=False)


def split_equal(ids: list, duration: list, num: int, min_duration: int = 256):

    sort = sorted(zip(duration, ids))[::-1]

    ids_split = [[] for i in range(num)]
    duration_spit = [[] for i in range(num)]
    duration_sum = [0] * num

    for d, i in sort:
        if d > min_duration:
            pos = np.argmin(duration_sum)

            ids_split[pos].append(i)
            duration_spit[pos].append(d)
            duration_sum[pos] =+ d

    return ids_split, duration_spit


def decode_vtt(content: str):
    '''
    :param content: String with the of the .vtt file.
    :return: String with combined text, List with Strings split at time stamps and List with float second time stamps.

    This Function decodes a vtt to get the contend with  time stamps.
    '''

    if '</c><' in content and '><c>' in content:

        # Split the content at line brake and check if word level time stamps are in the line.
        content = [l for l in content.split('\n') if '<c>' in l]

        # Connect list of strings back together.
        content = "".join(content)

        # Split String at time stamp headers.
        content = content.split('><c>')

        # Create output lists.
        words = []
        stamps = []

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
                words.append(' ' + word)
                stamps.append(stam)
            else:
                # If no time stamp contain in time stamp part we assume that it is a another word.
                # If it as a another word we add it to the previous word string.
                if len(words) > 0:
                    words[-1] = words[-1] + " " + c.replace('</c>', '').replace('<', '').lstrip().rstrip()

        return ''.join(words), words, stamps

    else:

        # Split the content at line brake.
        content = [l for l in content.split('\n')]

        # Create output lists.
        words_buffer = []
        stamps_buffer = []
        words = []
        stamps = []

        # Loop word and time stamp string list.
        for idx in range(len(content)):
            if ' --> ' in content[idx]:
                stamps_buffer.append(content[idx])

                word_buffer = []
                idx += 1
                while idx + 1 < len(content) and ' --> ' not in content[idx + 1]:
                    word_buffer.append(content[idx])
                    idx += 1

                words_buffer.append(" ".join(word_buffer))

        for idx in range(len(stamps_buffer)):
            s = stamps_buffer[idx].split(' --> ')

            s_1 = s[0]
            s_1 = s_1.split(':')
            s_1 = s_1[:-1] + s_1[-1].split('.')

            s_2 = s[1]
            s_2 = s_2.split(':')
            s_2 = s_2[:-1] + s_2[-1].split('.')

            s_1 = datetime.timedelta(hours=int(s_1[0]), minutes=int(s_1[1]), seconds=int(s_1[2]),
                                     milliseconds=int(s_1[3]))
            s_1 = s_1.total_seconds()

            s_2 = datetime.timedelta(hours=int(s_2[0]), minutes=int(s_2[1]), seconds=int(s_2[2]),
                                     milliseconds=int(s_2[3]))
            s_2 = s_2.total_seconds()

            stamps_buffer[idx] = [s_1, s_2]

        for idx in range(len(words_buffer)):
            word = words_buffer[idx].lstrip().rstrip()
            wor = [' ' + w for w in word.split(' ')]

            stamp = stamps_buffer[idx]

            time_snip = (stamp[1] - stamp[0]) / len(wor)

            stamps += [stamp[0] + i * time_snip for i in range(len(wor))]
            words += wor

        return ''.join(words), words, stamps



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


def char_level_encoder(words: list):

    chars = []

    for word in words:
        w = " " + word
        chars.append([ord(c) for c in w])

    return chars


def worker(work: list,
           save_dir,
           target_fps: int = 0,
           target_resolution: tuple = None,
           download: bool = True,
           lock = None,
           keep_buffer_download: bool = False,
           download_buffer_dir: str = '',
           use_subtitles: bool = False,
           tokenizer=None,
           language_tokens_per_frame: int = 4,
           padding_token: int = 50257,
           skip_if_no_subtitles: bool = True,
           service_account_json_path: str = '',
           bucket_name: str = '',
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
    :param lock: The multiprocessing lock used to control the youtube web site scraping.
    :param keep_buffer_download: If True the downloaded files will not be deleted after the tfrecord got created.
    :param download_buffer_dir: Directory where YoutubDL will download the videos (only if download is True (default)).
    It is recommended to use a RAM Disk as buffer directory.
    :param use_subtitles: Bool, if true Text will be used to.
    :param tokenizer: A lambda that contains the tokenizer.
    :param language_tokens_per_frame: The number of language tokens encoded per single frame.
    :param padding_token: The token id that is used as padding token.
    :param skip_if_no_subtitles: If True the video will be skipped if no subtitles are available.
    (only if use_subtitles is True)
    :param service_account_json_path: The path to the json containing the service account informations.
    :param bucket_name: The Name of the google cloud storage bucket the TFrecords are should to be uploaded to.
    :param youtube_base: Youtube base string https://www.youtube.com/watch?v=.

    This function will download youtube videos and proses them and save than as TF.record files.
    '''

    # Check if TF.record are uploaded to google cloud storage.
    if service_account_json_path != '':
        cloud_storage = True
        buffer_save_dir = download_buffer_dir

        cloud_storage_client = storage.Client.from_service_account_json(service_account_json_path)
        cloud_storage_bucket = cloud_storage_client.get_bucket(bucket_name)
    else:
        cloud_storage = False
        buffer_save_dir = save_dir

    # shuffle the work for better thread utilization.
    random.shuffle(work)

    # really small padding frame.
    pading_frame = cv2.imencode('.jpg', np.array([[[255]]]))[1].tobytes()

    # Check if video needs to be downloaded.
    if download:
        # Creat Youtube Downloader.
        youtube_getter = youtube_dl.YoutubeDL({'writeautomaticsub': True, 'ignore-errors': True, 'socket-timeout': 600})
        youtube_getter.add_default_info_extractors()

    # Loop to list of work.
    for wor in work:

        # Assume by default the download was successful.
        download_success = True

        # Assume by default the subtitles are available.
        subtitles_available = True

        video_buffer_path = ""
        video_urls = ""
        vtt_url = ""

        # Check if video needs to be downloaded.
        if download:

            # Execute all in try except to so the script is not crashing from a single fails video.
            try:

                print(youtube_base + wor)

                # We have to lock this part because it can lead to errors if multiple thread try to
                # scrap video Information at the same time.
                with lock:
                    # Get video info.
                    info = youtube_getter.extract_info(youtube_base + wor, download=False)

                # Go and find all video links that is as small as possible
                # and is still larger that the target resolution.
                if 'formats' in info:
                    current_res = (9999999, 9999999)

                    for f in info['formats']:

                        if 'format_note' in f:
                            if f['format_note'] != "tiny":
                                if 'width' in f and 'height' in f:
                                    width = f['width']
                                    height = f['height']

                                    if width is not None and height is not None:
                                        if width > target_resolution[0] and height > target_resolution[1]:
                                            if current_res[0] > width and current_res[1] > height:
                                                video_urls = []
                                                current_res = (width, height)

                                            if current_res[0] == width and current_res[1] == height:
                                                if 'ext' in f and 'url' in f:
                                                    video_urls.append(
                                                        {'width': width, 'height': height,
                                                         'ext': f['ext'], 'url': f['url']})

                # Go and find the english vtt subtitle link.
                if 'automatic_captions' in info:
                    automatic_captions = info['automatic_captions']
                    if 'en' in automatic_captions:
                        en_automatic_captions = automatic_captions['en']

                        for en in en_automatic_captions:
                            if 'ext' in en:
                                if en['ext'] == 'vtt':
                                    if 'url' in en:
                                        vtt_url = en['url']

                                    break

            except:
                download_success = False

            # Do some checking to ensure that the information get extracted successfully.
            if video_urls is None:
                download_success = False

            if not download_success and len(video_urls) == 0:
                download_success = False

            # Download the video and subtitle.
            if download_success:

                # Put .webm at the bottom at the list.
                for idx in range(len(video_urls)):
                    if video_urls[idx]['ext'] == 'webm':
                        video_urls[-1], video_urls[idx] = video_urls[idx], video_urls[-1]

                for video_url in video_urls:
                    url = video_url['url']
                    ext = video_url['ext']

                    if url is not None and ext is not None:
                        if url != "" and ext != "":
                            video_buffer_path = os.path.join(download_buffer_dir, wor) + '.' + ext
                            download_success = downloader(url, video_buffer_path)

                            if download_success:

                                # If no mp4 god downloaded yous ffmpag to converted it to mp4.
                                if ext != 'mp4':
                                    new_video_buffer_path = os.path.join(download_buffer_dir, wor) + '.mp4'

                                    subprocess.run(['ffmpeg', '-i', video_buffer_path, '-c',
                                                    'copy', new_video_buffer_path, '-y'],
                                                   capture_output=False, stdout=subprocess.DEVNULL,
                                                   stderr=subprocess.STDOUT)

                                    if os.path.exists(video_buffer_path):
                                        os.remove(video_buffer_path)

                                    video_buffer_path = new_video_buffer_path

                                # Check if the file can be opened.
                                video_cap = cv2.VideoCapture(video_buffer_path)
                                success, _ = video_cap.read()
                                video_cap.release()

                                if success:
                                    break
                                else:
                                    warnings.warn("cv2 failed to open:" + video_buffer_path)

                                    if os.path.exists(video_buffer_path):
                                        os.remove(video_buffer_path)



        else:
            vtt = os.path.splitext(wor)[0]
            vtt = os.path.join(download_buffer_dir, (vtt + '.*'))
            vtt = [v for v in glob.glob(vtt) if '.vtt' in v]

        # Download and tokenize teh vtt.
        if use_subtitles and download_success:

            vtt_download_success = False
            if vtt_url is not None:
                if vtt_url != "":
                    vtt = os.path.join(download_buffer_dir, wor) + '.vtt'
                    vtt_download_success = downloader(vtt_url, vtt)

            if vtt_download_success:
                try:
                    with open(vtt, "r") as r:
                        text, words, stamp = decode_vtt(r.read())
                except:
                    warnings.warn("Faild to open and decode: " + vtt)
                    vtt_download_success = False

            if vtt_download_success:
                bpe_list = tokenizer(words, text)

            else:
                subtitles_available = False
                bpe_list = []

        wor = video_buffer_path

        # Check if this video need to be skipt. This will happen if subtitles are required but are not available.
        if (not use_subtitles or \
           (subtitles_available or not skip_if_no_subtitles) or \
           (subtitles_available and skip_if_no_subtitles)) and download_success:

            # Setup CV2 video reader.
            video_cap = cv2.VideoCapture(wor)
            success, frame = video_cap.read()
            frame_count = 0

            # Get base video FPS, this is needed to align language and video.
            fps_raw = video_cap.get(cv2.CAP_PROP_FPS)

            # Calculate Modulo number.
            fps_split = diveision_zero(round(fps_raw), target_fps)

            if success:

                # Create TF Record Writer
                _save_name = os.path.join(buffer_save_dir, os.path.splitext(ntpath.basename(wor))[0]) + '.tfrecord'
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
                            frame = cv2.imencode('.jpg', frame)[1].tobytes()

                            if use_subtitles:
                                token_buffer = []
                                proto = []

                                while len(bpe_list) > 0 and stamp[0] < (frame_count + fps_split) * (1 / fps_raw):
                                    token_buffer += bpe_list[0]
                                    bpe_list.pop(0)
                                    stamp.pop(0)

                                for i in range(0, len(token_buffer), language_tokens_per_frame):
                                    buffer = token_buffer[i:i + language_tokens_per_frame]
                                    mask = language_tokens_per_frame - len(buffer)
                                    buffer += [padding_token] * mask
                                    skip_buffer = i > 0

                                    proto.append(frame_encoder(pading_frame if skip_buffer else frame,
                                                               buffer,
                                                               [skip_buffer], [mask]))

                                if len(proto) <= 0:
                                    proto.append(frame_encoder(frame,
                                                               [padding_token] * language_tokens_per_frame,
                                                               [False], [0]))

                            else:
                                # Encode frame to proto buffer.
                                proto = [frame_encoder(frame)]

                            # print(frame_count, len(proto))

                            # Write proto buffer to TF.record file.
                            for p in proto:
                                tf_writer.write(p)

                        # Get next frame.
                        success, frame = video_cap.read()
                        frame_count += 1

            # close video file.
            video_cap.release()
            success = True

        if cloud_storage and success:
            blob = cloud_storage_bucket.blob(os.path.join(save_dir,
                                                          os.path.splitext(ntpath.basename(wor))[0] + '.tfrecord'))
            if os.path.exists(_save_name):
                blob.upload_from_filename(_save_name)

        # Remove video file if it was downloaded.
        if download and not keep_buffer_download:
            if os.path.exists(wor):
                os.remove(wor)

            if use_subtitles and subtitles_available:
                if os.path.exists(vtt):
                    os.remove(vtt)

        if cloud_storage and not keep_buffer_download:
            if os.path.exists(_save_name):
                os.remove(_save_name)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('load_path', type=str,
                        help='The path to a json file containing video information, or a path to a folder containing '
                             'json files with video information.')
    parser.add_argument('save_path', type=str,
                        help='The path where the final TFrecords get saved.')
    parser.add_argument('download_buffer_path', type=str,
                        help="A Folder that gets used to save downloads. IF 'keep_buffer_download' is False (default) "
                             "the content in the folder will get automatically deleted when it is no longer needed.")

    parser.add_argument('-num_worker', type=int, default=1,
                        help='The number of parallel workers.')
    parser.add_argument('-download', type=str, default='True',
                        help='If the videos are supposed to be downloaded from youtube or if there should '
                             'be loaded from the buffer folder.')
    parser.add_argument('-use_subtitles', type=str, default='False',
                        help='If subtitles are supposed the be downloaded and encoded.')
    parser.add_argument('-skip_if_no_subtitles', type=str, default='True',
                        help='Skip the Video if no subtitles are available. If False the ')
    parser.add_argument('-keep_buffer_download', type=str, default='False',
                        help='If True the videos how god downloaded will be capt on disk.')

    parser.add_argument('-target_fps', type=int, default=1,
                        help='The FPS of the output TFrecord.')
    parser.add_argument('-target_width', type=int, default=320,
                        help='The frame width of the output TFrecord.')
    parser.add_argument('-target_height', type=int, default=176,
                        help='The frame height of the output TFrecord.')
    parser.add_argument('-language_tokens_per_frame', type=int, default=64,
                        help='The number of tokens aligned with a single frame.')
    parser.add_argument('-padding_token', type=int, default=0,
                        help="The token ID that gets usd to pad to the 'language_tokens_per_frame' number"
                             "if not enough tokens are present in the frame.")
    parser.add_argument('-duration_need_larger', type=int, default=256,
                        help='A single Video needs to be LONGER than this variable in seconds.')

    parser.add_argument('--service_account_json_path', type=str,
                        help="The path to the google service account information. If this is sat the TFrecords "
                             "will get automatically uploaded to a google cloud storage bucket."
                             " The bucket name needs to be defined with '--bucket_name'")
    parser.add_argument('--bucket_name', type=str,
                        help="The Name of the google cloud storage bucket the TFrecords are should to be uploaded to."
                             " This only as an affect if 'google service account information' are present.")


    args = parser.parse_args()

    load_path = args.load_path
    save_path = args.save_path
    download_buffer_path = args.download_buffer_path

    num_worker = args.num_worker
    download = str2bool(args.download)
    use_subtitles = str2bool(args.use_subtitles)
    skip_if_no_subtitles = str2bool(args.skip_if_no_subtitles)
    keep_buffer_download = str2bool(args.keep_buffer_download)

    target_fps = args.target_fps
    target_width = args.target_width
    target_height = args.target_height
    language_tokens_per_frame = args.language_tokens_per_frame
    padding_token = args.padding_token
    duration_need_larger = args.duration_need_larger

    service_account_json_path = args.service_account_json_path
    bucket_name = args.bucket_name

    assert os.path.exists(load_path), 'The load path is invalid.'
    assert download, 'Only download is supported at the moment.'

    if not os.path.exists(download_buffer_path):
        os.makedirs(download_buffer_path)

    if not os.path.exists(save_path) and service_account_json_path == '':
        os.makedirs(save_path)


    if os.listdir(load_path):
        load_path = [os.path.join(load_path, p) for p in os.listdir(load_path)]
    else:
        load_path = [load_path]

    ids = []
    duration = []

    for l in load_path:
        json_load = json.load(open(l))

        ids = ids + json_load['id']
        duration = duration + json_load['duration']

    ids, duration = split_equal(ids, duration, num_worker, duration_need_larger)

    split_video_count = 0
    split_video_duration = 0

    for i in range(len(ids)):
        buffer_video_count = len(ids[i])
        buffer_video_duration = sum(duration[i])

        print('split:', i, 'videos:', buffer_video_count, 'duration:', buffer_video_duration)

        split_video_count += buffer_video_count
        split_video_duration += buffer_video_duration

    print('total num of videos:', split_video_count, 'total video duration:', split_video_duration)


    youtube_getter = youtube_dl.YoutubeDL({'writeautomaticsub': True, 'ignore-errors': True, 'socket-timeout': 600})
    youtube_getter.add_default_info_extractors()

    lock = multiprocessing.Lock()

    worker_list = []

    for work in ids:

        p = multiprocessing.Process(target=worker, args=(work,
                                                         save_path,
                                                         target_fps,
                                                         (target_width, target_height),
                                                         download,
                                                         lock,
                                                         keep_buffer_download,
                                                         download_buffer_path,
                                                         use_subtitles,
                                                         lambda words, text: char_level_encoder(words),
                                                         language_tokens_per_frame,
                                                         padding_token,
                                                         skip_if_no_subtitles,
                                                         service_account_json_path,
                                                         bucket_name))

        p.start()
        worker_list.append(p)

    for w in worker_list:
        w.join()

    if download and not keep_buffer_download:
        os.remove(download_buffer_path)

    exit()


