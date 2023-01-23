#!/usr/bin/env bash
for seed in 12 22 32 42 52
do
    for word_input in true false
    do
        for prompt in {1..8}
        do
            python CV_CTS.py --prompt_id ${prompt} --seed ${seed} --word_input ${word_input}
        done
    done
done