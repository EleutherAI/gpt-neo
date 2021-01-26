from functools import partial
import typing

from src.dataclass import ModelParameter
from .inputs import dataset
from src.utils_core import chunks

from tensorflow_estimator.python.estimator import estimator as estimator_lib
import scipy.ndimage
import numpy as np
import cv2


def render_video(model_output: typing.List[typing.Tuple[np.ndarray, typing.List[str]]],
                 count: int,
                 params: ModelParameter,
                 save_prefix: str = "",
                 upscale: int = 4,
                 line_split: int = 2,
                 text_color: typing.Tuple[int, int, int] = (255, 0, 255),
                 text_pos: typing.Tuple[int, int] = (10, 625),
                 text_size: float = 1.27,
                 text_thickness: int = 3,
                 text_line_offset: int = 50):

    writer = cv2.VideoWriter(f"{save_prefix}_{count}.avi", cv2.VideoWriter_fourcc(*"MJPG"), 1,
                             (params.frame_width * upscale * len(model_output), params.frame_height * upscale))

    for idx in range(len(model_output[0][0])):
        frame = []
        for sub_idx in range(len(model_output)):

            sub_frame = model_output[sub_idx][0][idx]
            sub_frame = sub_frame * 255
            sub_frame = scipy.ndimage.zoom(sub_frame, (upscale, upscale, 1), order=0)
            sub_frame = np.uint8(sub_frame)
            cv2.cvtColor(sub_frame, cv2.COLOR_RGB2BGR)

            text = model_output[sub_idx][1]
            if text is not None:
                for i, _text in enumerate(chunks(text[idx], params.language_token_per_frame // line_split)):

                    cv2.putText(sub_frame, _text, (text_pos[0], text_pos[1] + text_line_offset * i),
                                cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, text_thickness)

            frame.append(sub_frame)

        frame = np.concatenate(frame, axis=1)
        writer.write(frame)

    writer.release()


def process_token_output(token_out: np.ndarray, padding_token: int, do_argmax: bool = True) -> typing.List[str]:

    _shape = token_out.shape
    if do_argmax:
        token_out = np.reshape(token_out, newshape=(_shape[0], _shape[1] * _shape[2], _shape[3]))
        token_out = np.argmax(token_out, axis=2)
    else:
        token_out = np.reshape(token_out, newshape=(_shape[0], _shape[1] * _shape[2]))

    token_out_str = []

    for token in token_out:
        if padding_token in token:
            token = token[:token.tolist().index(padding_token)]

        token_out_str.append("".join([chr(tok) if tok > 31 and tok != 127 else " " for tok in token]))

    return token_out_str


def process_video_output(out_frame: np.ndarray, params: ModelParameter) -> np.ndarray:

    out_frame = np.reshape(out_frame, (params.time_patch_size, params.frame_height_patch, params.frame_width_patch,
                                       params.time_patch, params.patch_size, params.patch_size, params.color_channels))

    out_frame = np.transpose(out_frame, [0, 3, 1, 4, 2, 5, 6])
    out_frame = np.reshape(out_frame, (params.n_ctx, params.frame_height, params.frame_width, 3))

    return out_frame


def gen_sample(estimator: estimator_lib, params: ModelParameter):

    pred_input_fn = partial(dataset, step=0)
    predict_estimator = estimator.predict(input_fn=pred_input_fn)

    for sample_idx in range(params.num_of_sample):
        out = next(predict_estimator)
        print('sample_idx:', sample_idx)

        frame_out = out['frame_out']

        frame_out = process_video_output(frame_out, params)

        if params.use_language:
            token_out = process_token_output(out['token_out'], params.padding_token)
        else:
            token_out = None

        render_input = []

        if not params.use_autoregressive_sampling:
            frame_inp = out['frame_inp']
            frame_inp = frame_inp[1:params.time_patch_size + 1]
            frame_inp = process_video_output(frame_inp, params)

            if params.use_language:
                token_inp = process_token_output(out['token_inp'], params.padding_token, False)
            else:
                token_inp = None

            render_input.append((frame_inp, token_inp))

        render_input.append((frame_out, token_out))

        render_video(render_input, sample_idx, params)
