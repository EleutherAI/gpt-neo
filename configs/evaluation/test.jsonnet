local models = import '../models.libsonnet';
local datasets = import '../datasets.libsonnet';

local dataset = datasets.RandomSumGenerator(examples_to_gen=1000, seed=1337);
local infeed = models.InFeed();

{
    model: models.GPT2(),
    model_path: "/tmp/checkpoints/",
    infeed: infeed,
    schedule: {
        steps:  dataset.examples_count
    },
}