import json


def find_consecutive_subset(numbers, x, no_start):
    numbers = sorted(list(set(numbers)))

    for i in range(len(numbers) - x + 1):
        sequence = numbers[i : i + x]

        if sequence[0] in no_start or sequence[-1] in no_start:
            continue

        if sequence == list(range(sequence[0], sequence[0] + x)):
            return sequence

    return None  # Return None if no consecutive subset is found


def get_processed_instances(output_file):
    processed_ids = set()
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    instance = json.loads(line.strip())
                    processed_ids.add(instance["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    except FileNotFoundError:
        pass
    return processed_ids
