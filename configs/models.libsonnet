local Dataset() = {
    kind: 'tfrecord',
    sources: ['/tmp/uds-preprocess/*.tfrecord'],
};
local vocab_size = 128;
{

    GPT2() :: {
        type: "GPT2",
        n_ctx: 8,
        n_embd: 8,
        n_head: 8,
        n_vocab: vocab_size,
        n_layer: 2,
        scale_by_depth: false,
        scale_by_in: false,
        mesh_shape: "batch:1",
        layout: "batch:1",
        activation_function: "gelu",
        attention_types: [
            [["global"], self.n_layer],
        ],
        auto_layout: false,
        auto_layout_and_mesh_shape: false,
    },

    InFeed() :: {
        batch_size: 8,
        random: {
            context_length: 8,
            vocab_size: vocab_size,
        },
        dataset: Dataset(),
    }
}