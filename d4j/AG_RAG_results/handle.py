def main():
    result = []
    for i in range(1, 11):
        with open(f"{i}.txt", 'r') as file:
            lines = [line.strip() for line in file.readlines()]
            for line in lines:
                result.append(line)

    print(len(set(result)))


if __name__ == '__main__':
    main()
