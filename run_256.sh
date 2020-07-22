bash make_hosts_wrapper
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/lib
while true; do
  python3 main.py --tpu isaac --model configs/gpt2_256.json --steps_per_checkpoint 3000
  echo "Recreating isaac in 60sec..."
  sleep 60
  pu recreate isaac --yes
done