local datasets = import '../datasets.libsonnet';

{
    name: "add-one-seq2seq",
    description: "sequence to sequence",

    dataset: datasets.RandomSumGenerator()
}