for b in 100 500 1000 2000
do
    for r in 0.1 0.15 0.2
    do 
        echo "b=" $b "and r=" $r
        python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b $b -lr $r -rtg --exp_name q2_b"$b"_r"$r"
    done
done