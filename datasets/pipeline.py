import ftfy
import tensorflow as tf

class BufferedEncodedStream(object):
    """
    Loads a file into memory, optionally fixes unicode, encodes it and adds the seperator to the beginning
    If set to text_mode the input is assumed to not be a file but the direct string data
    """
    def __init__(self, inp, encoder, seperator=None, fix=False, minimum_size=0, text_mode=False):
        if text_mode:
            d = inp
        else:
            with open(inp, "r") as f:
                d = f.read()

        if fix:
            d = ftfy.fix_text(d, normalization='NFKC')
        self.data = encoder.encode(d).ids
        
        if len(self.data) < minimum_size or all([x == 0 for x in self.data]): # Sanity check
            self.data = [] # Don't return file contents if it doesn't pass the sanity check
        elif seperator is not None: # Only add seperator if sanity check didn't fail
            self.data = seperator + self.data # Seperator should be [tokens]
        
        self.idx = 0
        self.n = len(self.data)

    def read(self, size=None):
        if self.idx < self.n:
            if size is None or size < 0:
                chunk = self.data[self.idx:]
                self.idx = self.n
            else:
                chunk = self.data[self.idx:self.idx + size]
                self.idx += len(chunk)
            return chunk
        else:
            return []

class EncodedCompressedReader:
    # Loads, encodes and concatenates the texts within a zst archive
    # Pass a archive, returns a stream of its concatenated contents
    def __init__(self, f, encoder, seperator=None, fix=False, minimum_size=0):
        def _gen():
            # Generator yielding the files inside the archive as encoded streams
            g = Reader(f).stream_data()
            for s in g:
                yield BufferedEncodedStream(s, encoder, seperator, fix, minimum_size, text_mode=True)
                
        self.g = _gen()

        try:
            self.current = next(self.g) # Current is, whenever possible, pointing to the currently opened stream in the archive
        except StopIteration:
            self.current = None
    
    def read(self, size=None):
        if size < 0:
            size = None
        remaining = size
        data = []

        while self.current and (remaining > 0 or remaining is None):
            data_read = self.current.read(remaining or -1)
            if len(data_read) < remaining or remaining is None: # Exhausted file
                try:
                    self.current = next(self.g)
                except StopIteration:
                    self.current = None
            if not remaining is None:
                remaining -= len(data_read)
            data.extend(data_read)
        
        return data

class EncodedConcatenatedFiles(object):
    """ 
    Stitches list of file names into a stream of properly encoded and seperated tokens
    Pass in a list of files and it outputs a stream of their contents stitched together
    """
    def __init__(self, fns, encoder, seperator=None, fix=False, minimum_size=0): 
        self.fs = list(reversed(fns)) # reversed because read() reads from the last element first
        self.enc = encoder
        self.seperator = seperator # Seperator should be [tokens]
        self.fix = fix
        self.minimum_size = minimum_size

    def read(self, size=None):
        if size < 0:
            size = None
        remaining = size
        data = []

        while self.fs and (remaining > 0 or remaining is None):
            if isinstance(self.fs[-1], str): # If the last element in the list is a string, it's an unopened file
                if self.fs[-1].endswith(".zst") or self.fs[-1].endswith(".xz"): # If it's an archive, we use this reader
                    self.fs[-1] = EncodedCompressedReader(self.fs[-1], self.enc, self.seperator, self.fix, self.minimum_size)
                else: # Otherwise we assume it's a normal text file
                    self.fs[-1] = BufferedEncodedStream(self.fs[-1], self.enc, self.seperator, self.fix, self.minimum_size)

            data_read = self.fs[-1].read(remaining or -1)
            if len(data_read) < remaining or remaining is None: # If we exhaust the file we're reading, pop it off the list
                self.fs.pop()

            if not remaining is None:
                remaining -= len(data_read)
            data.extend(data_read)

        return np.array(data, np.int32)

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_example(example_proto, max_seq_len=1024) -> dict:
    features = {
        # "id": tf.FixedLenFeature([1], tf.int64),
        "content": tf.FixedLenFeature([max_seq_len], tf.int64)
    }
    return tf.parse_single_example(example_proto, features)

def create_example(eid, data) -> tf.train.Example:
    feature = {
        # "id": _int64_feature([eid % math.max64]),
        "content": _int64_feature(data)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))