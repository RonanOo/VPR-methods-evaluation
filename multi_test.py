from mult_main import test
import parser
import os
import pickle


if __name__ == "__main__":
    args = parser.parse_arguments()

    # Variable arguments
    model_files = [
        "models/GCL/" + file
        for file in os.listdir("models/GCL")
        if file.endswith(".pth")
    ]
    datasets = [
        f.path + "/images/test/"
        for f in os.scandir("../VPR-datasets-downloader/datasets/")
        if f.is_dir()
    ]
    if "san_francisco" in datasets:
        datasets.remove("san_francisco")

    # Limit models to the numbers specified in the arguments
    model_files = model_files[(args.number * 6) : ((args.number + 1) * 6)]

    final_dictionary = {}

    for dataset in datasets:
        for model_file in model_files:
            result = test(
                args.positive_dist_threshold,
                args.method,
                args.backbone,
                args.descriptors_dimension,
                dataset + "database",
                dataset + "queries",
                args.num_workers,
                args.batch_size,
                args.exp_name,
                args.device,
                args.recall_values,
                args.num_preds_to_save,
                args.save_only_wrong_preds,
                model_file,
            )
            if dataset not in result:
                final_dictionary[dataset] = [(model_file, result)]
            else:
                final_dictionary[dataset] = final_dictionary[dataset] + [
                    (model_file, result)
                ]
            break

    with open("results/" + str(args.number) + ".pkl", "wb") as f:
        pickle.dump(final_dictionary, f)
