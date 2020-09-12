PORT=${1:-8080}
tensorboard --logdir ./.test/gpt_test --host 0.0.0.0 --reload_multifile True --port $PORT
