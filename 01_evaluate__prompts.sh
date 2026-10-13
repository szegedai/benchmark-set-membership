settings=(
    natural_1_2
    natural_3_4
    coding_simple
    coding_algorithmic
)

sets=(
    related_words_strong_directions
    related_words_strong_subdirections
    related_words_strong_students
    related_words_strong_houses
    related_words_strong_elements
    related_words_strong_matter
    related_words_strong_teeth
    related_words_strong_operations
    related_words_strong_cards
    related_words_strong_dna
    related_words_loose_animals
    related_words_loose_fruits
    related_words_loose_clothes
    unrelated_words1
    unrelated_words2
    unrelated_words3
    related_numbers_d2
    related_numbers_d75
    related_numbers_d100
    unrelated_numbers1
    unrelated_numbers2
    unrelated_numbers3
)

models=(
    meta-llama/Llama-3.2-3B-Instruct
    meta-llama/Llama-3.1-8B-Instruct
    mistralai/Mistral-Small-24B-Instruct-2501
    Qwen/Qwen2.5-32B-Instruct
    microsoft/Phi-3.5-MoE-instruct
    meta-llama/Llama-3.1-70B-Instruct
    meta-llama/Llama-3.3-70B-Instruct
)

for model in "${models[@]}"
do
    for setting in "${settings[@]}"
    do
        for set in "${sets[@]}"
        do
            python 01_evaluate__prompts.py \
                --setting $setting \
                --model $model \
                --set $set
        done
    done
done
