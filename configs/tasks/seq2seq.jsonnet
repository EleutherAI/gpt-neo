{
    name: "seq2seq",
    description: "sequence to sequence",
    special_tokens : {
        PAD: 0,
        EOS: 1,
        BOS: 2,
    },
    vocab_size: 256 - len(self.special_tokens),
}