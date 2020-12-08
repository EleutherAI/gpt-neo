import tensorflow as tf
import os
import glob


total_files = 0
total_tokens = 0
empty_count = 0
nums = []
for in_file in sorted(glob.glob("/home/connor/sid/fix_inputs/tfrecords/*")):
        n_examples = 0
        for example in tf.python_io.tf_record_iterator(in_file):
            n_examples += 1
            x = tf.train.Example.FromString(example)
            y = x.features.feature["text"].int64_list
            nums.extend(y.value)
            # print(len(y.value))
            total_tokens += len(y.value)
            total_files += 1
            if len(y.value) == 0:
                    empty_count += 1
        print(n_examples, in_file)
print(sorted(nums))
# print(total_tokens)
# print(total_files)
# print(empty_count)
# print(f"du -sh {in_file}: ")
# os.system(f"du -sh {in_file}")



