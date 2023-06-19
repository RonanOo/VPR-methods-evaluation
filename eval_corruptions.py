import csv

import argparse

corruptions = [
    "shot_noise",
    "defocus_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "elastic_transform",
    "jpeg_compression",
    "rotate",
    "crop",
]

class TestParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--subset', required=False, default='val', help='For MSLS. Subset to evaluate')
        self.parser.add_argument("--models", required=True, nargs="+", help='Models to be evaluated')
        self.parser.add_argument("--base", required=False, default=None,
                                 help='Model that will be used as a base model. The first model in --models will be used if this parameter is None')

    def parse(self):
        self.opt = self.parser.parse_args()


def readCorruptData(models, subset):
    res = {}
    for model in models:
        modelRes = {}
        for corruption in corruptions:
            corruptionRes = {}
            for severity in range(1, 6):
                severityRes = {}
                if "resnet50_avg" in model:
                    severityFile = f"results/msls/{subset}/{model}/{model}_{corruption}_{severity}_result.txt"
                else:
                    severityFile = f"results/msls/{subset}/{model}/MSLS_resnet152_GeM_480_GCL_{corruption}_{severity}_result.txt"
                with open(severityFile, 'r') as f:
                    for line in f.readlines():
                        split = line.split(": ")
                        severityRes[split[0]] = float(split[1])
                corruptionRes[severity] = severityRes
            modelRes[corruption] = corruptionRes
        res[model] = modelRes
    return res


def readCleanData(models, subset):
    res = {}
    for model in models:
        modelRes = {}
        if "resnet50_avg" in model:
            with open(f"results/msls/{subset}/{model}/{model}_result.txt", 'r') as f:
                for line in f.readlines():
                    split = line.split(": ")
                    modelRes[split[0]] = float(split[1])
        else:
            with open(f"results/msls/{subset}/{model}/MSLS_resnet152_GeM_480_GCL_result.txt", 'r') as f:
                for line in f.readlines():
                    split = line.split(": ")
                    modelRes[split[0]] = float(split[1])
        res[model] = modelRes
    return res


def resultsToCsv(results, subset, filename):
    csv_columns = list(list(results.values())[0].keys())
    with open(f"results/msls/{subset}/{filename}", 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in results.values():
            writer.writerow(data)


if __name__ == "__main__":
    # these are left here for easy copy pasting
    # MSLS_resnet50_avg_480_CL
    # MSLS_resnet50_GeM_480_CL
    # MSLS_resnet50_avg_480_GCL
    # MSLS_resnet50_GeM_480_GCL
    # MSLS_resnet152_avg_480_CL
    # MSLS_resnet152_GeM_480_CL
    # MSLS_resnet152_avg_480_GCL
    # MSLS_resnet152_GeM_480_GCL
    # MSLS_resnext_avg_480_CL
    # MSLS_resnext_GeM_480_CL
    # MSLS_resnext_avg_480_GCL
    # MSLS_resnext_GeM_480_GCL

    p = TestParser()
    p.parse()
    params = p.opt

    models = params.models
    subset = params.subset
    
    print(corruptions)

    if not params.base:
        base = models[0]
    else:
        base = params.base

    corruptData = readCorruptData(models, subset)
    cleanData = readCleanData(models, subset)

    #################################################### CORRUPT RECALL ####################################################
    corruptRecall1s = {
        model: {
            corruption: {
                severity: corruptData[model][corruption][severity]['all_recall@1']
                for severity in range(1, 6)
            } for corruption in corruptData[model]
        } for model in corruptData
    }

    corruptRecallAt1 = {
        model: {
            corruption: sum(corruptRecall1s[model][corruption].values())
                        / sum(corruptRecall1s[base][corruption].values())
            for corruption in corruptData[model]
        } for model in corruptData
    }

    # add model and mean CR@1 to the dict
    finalCorruptRecallAt1 = {model: {
        'model': model,
        'R@1': "{:.2f}".format(cleanData[model]['all_recall@1']),
        'mCR@1': "{:.2f}".format(sum(corruptRecallAt1[model].values()) / len(corruptions))
    } for model in models}

    for model in models:
        for corruption in corruptions:
            finalCorruptRecallAt1[model][corruption] = "{:.2f}".format(corruptRecallAt1[model][corruption])

    resultsToCsv(finalCorruptRecallAt1, subset, 'meanCorruptRecall@1.csv')
    print(finalCorruptRecallAt1)

    #################################################### RELATIVE CORRUPT RECALL ####################################################
    relativeCorruptRecallAt1 = {
        model: {
            corruption: (sum(corruptRecall1s[model][corruption].values()) - 5 * cleanData[model]['all_recall@1'])
                        / (sum(corruptRecall1s[base][corruption].values()) - 5 * cleanData[base]['all_recall@1'])
            for corruption in corruptData[model]
        } for model in corruptData
    }

    # add model and mean CR@1 to the dict
    finalRelativeCorruptRecallAt1 = {model: {
        'model': model,
        'R@1': "{:.2f}".format(cleanData[model]['all_recall@1'] / cleanData[base]['all_recall@1']),
        'mCR@1': "{:.2f}".format(sum(corruptRecallAt1[model].values()) / len(corruptions)),
        'rel. mCR@1': "{:.2f}".format(sum(relativeCorruptRecallAt1[model].values()) / len(corruptions))
    } for model in models}

    for model in models:
        for corruption in corruptions:
            finalRelativeCorruptRecallAt1[model][corruption] = "{:.2f}".format(
                relativeCorruptRecallAt1[model][corruption])

    resultsToCsv(finalRelativeCorruptRecallAt1, subset, 'meanRelativeCorruptRecall@1.csv')
    print(finalRelativeCorruptRecallAt1)

    #################################################### ALL RECALLS ####################################################
    recalls = [
        'all_recall@1',
        'all_recall@5',
        'all_recall@10',
        'all_recall@20',
        'all_map@1',
        'all_map@5',
        'all_map@10',
        'all_map@20',
    ]
    allResults = {
        model: {
            recall:
                sum([
                    sum([corruptData[model][corruption][severity][recall] for severity in range(1, 6)]) /
                    sum([corruptData[base][corruption][severity][recall] for severity in range(1, 6)])
                    for corruption in corruptions]) / len(corruptions)
            for recall in recalls
        } for model in corruptData
    }
    finalAllResults = {model: {
        'model': model,
    } for model in models}
    for model in models:
        for recall in recalls:
            finalAllResults[model][recall] = "{:.2f}".format(allResults[model][recall])

    resultsToCsv(finalAllResults, subset, 'allResults.csv')
    print(finalAllResults)
