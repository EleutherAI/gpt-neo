bash make_hosts_wrapper
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/lib
while true; do
  python3 main.py --tpu sanchopanza --model configs/v8_test.json --steps_per_checkpoint 100
  echo "Recreating sanchopanza in 60sec..."
  sleep 60
  pu recreate sanchopanza --yes
done