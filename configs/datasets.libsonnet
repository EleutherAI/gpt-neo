{
    RandomSumGenerator(examples_to_gen=1000, 
                       context_length=256, 
                       seed=1337,
                       vocab_size=256) :: {
        kind: 'datasets.RandomSumGenerator',
        seed: seed,
        context_length: context_length,
        examples_count: examples_to_gen,
        vocab_size: vocab_size,
        special_tokens : {
            PAD: 0,
            EOS: 1,
            BOS: 2,
        } 
    },
    
    Seq2SeqTFRecordDataset(location="", n_samples=0, vocab_size=0, context_length=0) :: {
        kind: 'datasets.Seq2SeqTFRecordDataset',
        context_length: context_length,
        n_samples: n_samples,
        vocab_size: vocab_size,
    }
}