import pandas as pd


def main():
    with open("../result/NewDataSet/ATLAS_Result.txt", 'r') as result_file, open("../result/NewDataSet/ATLAS_Label.txt",
                                                                                 'r') as label_file:
        results = [line.strip() for line in result_file.readlines()]
        labels = [line.strip() for line in label_file.readlines()]
        label_names = ["label" + str(i) for i in range(1, len(labels) + 1)]
        atlas_correct = [int(result == label) for result, label in zip(results, labels)]
        print(f"atlas accuracy: {sum(atlas_correct) / len(atlas_correct)}")

    with open("../result/NewDataSet/EditAS_Result.txt", 'r') as result_file, open(
            "../result/NewDataSet/EditAS_Label.txt",
            'r') as label_file:
        results = [line.strip() for line in result_file.readlines()]
        labels = [line.strip() for line in label_file.readlines()]
        editas_correct = [int(result == label) for result, label in zip(results, labels)]
        print(f"EditAS accuracy: {sum(editas_correct) / len(editas_correct)}")

    with open("../result/NewDataSet/IR_Result.txt", 'r') as result_file, open("../result/NewDataSet/IR_Label.txt",
                                                                              'r') as label_file:
        results = [line.strip() for line in result_file.readlines()]
        labels = [line.strip() for line in label_file.readlines()]
        ir_correct = [int(result == label) for result, label in zip(results, labels)]
        print(f"IR accuracy: {sum(ir_correct) / len(ir_correct)}")

    with open("../result/NewDataSet/AG-RAG_Result.txt", 'r') as result_file, open("../result/NewDataSet/AG-RAG_Label.txt",
                                                                           'r') as label_file:
        results = [line.strip() for line in result_file.readlines()]
        labels = [line.strip() for line in label_file.readlines()]
        our_correct = [int(result == label) for result, label in zip(results, labels)]
        print(f"our accuracy: {sum(our_correct) / len(our_correct)}")

    with open("../result/NewDataSet/Integration_Result.txt", 'r') as result_file, open("../result/NewDataSet/Integration_Label.txt", 'r') as label_file:
        results = [line.strip() for line in result_file.readlines()]
        labels = [line.strip() for line in label_file.readlines()]
        integration_correct = [int(result == label) for result, label in zip(results, labels)]
        print(f"Integration accuracy: {sum(integration_correct) / len(integration_correct)}")

    pd.DataFrame({
        "labels": label_names,
        'ATLAS': atlas_correct,
        "EditAS": editas_correct,
        "Integration": integration_correct,
        "IR": ir_correct,
        "Ours": our_correct
    }).to_excel("./baselines_new.xlsx", index=False)


if __name__ == '__main__':
    main()
