from datasets import load_dataset

from logging import getLogger

logger = getLogger(__name__)

DURATION_LB = 0.5
DURATION_UB = 30


def audio_duration(x):
    audio = x["audio"]

    return len(audio["array"]) / audio["sampling_rate"]

def filter_duration(x):
    return DURATION_LB <= audio_duration(x) <= DURATION_UB


def filter_data(x):
    return filter_duration(x)


def make_dataset_small(dataset):
    count = {
        "male": 0,
        "female": 0
    }
    N = 5
    male_indices = []
    female_indices = []
    for i, x in enumerate(dataset):
        if count["male"] == N and count["female"] == N:
            break
        if x["gender"] == "male" and count["male"] < N:
            male_indices.append(i)
            count["male"] += 1
        if x["gender"] == "female" and count["female"] < N:
            female_indices.append(i)
            count["female"] += 1

    indices = male_indices + female_indices

    dataset_small = dataset.select(indices)
    return dataset_small


if __name__ == "__main__":
    dataset = load_dataset("alexandrainst/coral-tts", split="train")

    gender_column = ["male" if x == "mic" else "female" for x in dataset["speaker_id"]]
    dataset = dataset.add_column("gender", gender_column)

    dataset_filtered = dataset.filter(filter_data)

    dataset_filtered_small = make_dataset_small(dataset=dataset_filtered)

    if True:
        dataset_filtered.push_to_hub("oliverkinch/coral-tts-filtered", private=True)
        dataset_filtered_small.push_to_hub("oliverkinch/coral-tts-filtered-small", private=True)
        

