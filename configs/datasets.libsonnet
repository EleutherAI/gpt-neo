{
    RandomSumGenerator(examples_to_gen=1000, seed=1337) :: {
        kind: 'datasets.RandomSumGenerator',
        seed: seed,
        examples_count: examples_to_gen,
        vocab_size: 64000,
    },
}