#!/usr/bin/env bash
for seed in 12 22 32 42 52
do
    for input_seq in 'word' 'pos'
    do
        for prompt in {1..8}
        do
            python CV_CTS.py --prompt_id ${prompt} --seed ${seed} --input ${input_seq}
        done
    done
done