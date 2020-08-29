import fasttext
import os
import sys
from absl import app, logging, flags
"""
Filter lines of text by language using fasttext
"""
FLAGS = flags.FLAGS
# https://fasttext.cc/docs/en/language-identification.html
VALID_LANGS ='af als am an ar arz as ast av az azb ba bar bcl be bg bh bn bo bpy br bs bxr ca cbk ce ceb ckb co cs cv cy da de diq dsb dty dv el eml en eo es et eu fa fi fr frr fy ga gd gl gn gom gu gv he hi hif hr hsb ht hu hy ia id ie ilo io is it ja jbo jv ka kk km kn ko krc ku kv kw ky la lb lez li lmo lo lrc lt lv mai mg mhr min mk ml mn mr mrj ms mt mwl my myv mzn nah nap nds ne new nl nn no oc or os pa pam pfl pl pms pnb ps pt qu rm ro ru rue sa sah sc scn sco sd sh si sk sl so sq sr su sv sw ta te tg th tk tl tr tt tyv ug uk ur uz vec vep vi vls vo wa war wuu xal xmf yi yo yue zh'.split()

flags.DEFINE_multi_enum('lang', enum_values=VALID_LANGS, default='en', help='language to include. can be specified multiple time for multi language')
flags.DEFINE_float('threshold', default=0.9, help='language to include. can be specified multiple time for multi language')
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
flags.DEFINE_boolean('reverse', False, 'reverse the filter')

ext = '.bin'
if not os.path.exists('lid.176' + ext): 
    logging.info('downloading model')
    os.system('wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176' + ext)
    logging.info('downloading model: done')

lid_model = fasttext.load_model('lid.176' + ext)

def main(_):
    while True:
        line = sys.stdin.readline()
        if not line: break
        
        (labels, scores)= lid_model.predict(line.strip())
        lang = labels[0].split("__label__")[1]
        score = scores[0]
        test = lang in FLAGS.lang and score > FLAGS.threshold

        if FLAGS.reverse and not test:
            sys.stdout.write(line)
        elif not FLAGS.reverse and test:
            sys.stdout.write(line)
        elif FLAGS.debug:
            logging.info(str(score, lang, FLAGS.lang, FLAGS.threshold))

if __name__ == '__main__':
    app.run(main)
