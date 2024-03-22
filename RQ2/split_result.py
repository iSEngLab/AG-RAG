import sys

def format_data(file_path: str, gold_path: str, output_path: str) -> None:
    with open(file_path, 'r') as origin_file, open(gold_path, 'w') as gold_file, open(output_path, 'w') as output_file:
        lines = [line.strip() for line in origin_file.readlines()]
        targets = []
        predictions = []

        for i in range(0, len(lines) - 1):
            line = lines[i]
            if line == 'target:':
                targets.append(lines[i + 1])
            elif line == 'raw_predictions:':
                predictions.append(lines[i + 1])

        assert len(targets) == len(predictions)
        for i in range(len(targets)):
            gold_file.write(f"{i}\t{targets[i]}\n")
            output_file.write(f"{i}\t{predictions[i]}\n")


def main():
    file_path = sys.argv[1]
    gold_path = sys.argv[2]
    output_path = sys.argv[3]
    format_data(file_path, gold_path, output_path)


if __name__ == '__main__':
    main()
