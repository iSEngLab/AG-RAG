import pandas as pd
import argparse
import re


def get_assertion_type(args):
    assertion_types = ['Equals', 'True', 'That', 'NotNull', 'False', 'Null', 'ArrayEquals', 'Same']
    assertion_types_with_assert = ['assert' + i for i in assertion_types]
    df = pd.read_csv(args.data_file)
    source = df['source']
    target = df['target']
    match = []
    with open(args.result_file, 'r', encoding="utf-8") as result_file, open(args.gold_file, 'r',
                                                                            encoding="utf-8") as gold_file:
        results = [line.split("\t")[1].strip() for line in result_file.readlines()]
        golds = [line.split("\t")[1].strip() for line in gold_file.readlines()]
        for result, gold in zip(results, golds):
            match.append(int(result == gold))
    # # 构建一个二维列表来统计对应的单元格
    table_data = [[[0, 0] for _ in range(9)] for _ in range(1)]

    for i in range(len(source)):
        t = target[i]
        m1 = match[i]
        row = 8

        # find assert, start with 'assert' or 'Assert . assert'
        if args.dataset == 'new':
            re_match = re.search(r'\bassert\w*', t)
            if re_match:
                assertion = re_match.group()
                if assertion in assertion_types_with_assert:
                    row = assertion_types_with_assert.index(assertion)
        elif args.dataset == 'old':
            n = t.split(" ")[6]
            if n in assertion_types_with_assert:
                row = assertion_types_with_assert.index(n)
            else:
                print(f"{i}, {n}")

        m = [m1]
        for j in range(1):
            table_data[j][row][0] += 1
            table_data[j][row][1] += m[j]

    nums = []
    accus = []
    for i in range(1):
        model_accu = 0
        for j in range(9):
            accu = table_data[i][j][1] / table_data[i][j][0]
            num = table_data[i][j][1]
            nums.append(num)
            accus.append(round(accu * 100, 2))
            model_accu += num
            table_data[i][j] = '{} ({}%)'.format(num, round(accu * 100, 2))
        print(model_accu)
        print(model_accu / len(source))

    # 将统计结果转换为DataFrame
    columns_labels = assertion_types
    columns_labels.append('Other')
    index_labels = ['Joint_new'] if args.dataset == 'new' else ['Joint_old']
    df = pd.DataFrame(table_data, index=index_labels, columns=columns_labels)
    print(df)
    df.to_csv(args.output_file, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, help="dataset file")
    parser.add_argument("--result_file", type=str, help="result file")
    parser.add_argument("--gold_file", type=str, help="gold file")
    parser.add_argument("--output_file", type=str, help="assert type output result file")
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    get_assertion_type(args)
