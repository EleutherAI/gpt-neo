bash make_hosts_wrapper
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/lib
while true; do
  python3 main.py --tpu pile --model_params configs/gpt2_256_nv.json --steps_per_checkpoint 3000 --autostack
  echo "Recreating isaac in 60sec..."
  sleep 60
  pu recreate pile --yes
done