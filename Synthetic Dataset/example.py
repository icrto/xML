from SyntDataset import Dataset

# creating dataset
dataset = Dataset(config_file="config.json")
if dataset.create_dataset() != -1:
    # saving & importing dataset
    dtset = dataset.import_dataset()
    if dtset is not None:
        dataset.save("data_simplified.txt", dtset)
        data = dataset.load("data_simplified.txt")
        y = [1 if int(datum["exists"]) > 0 else 0 for datum in data]
        print("Positive class: %d images" % y.count(1))
        print("Negative class: %d images" % y.count(0))
        pos_class = [datum for datum in data if int(datum["exists"]) > 0]
        bndboxes = dataset.get_bndboxes(pos_class[1])
        dataset.display_img(pos_class[1]["img"], bndboxes)
    else:
        print("Error loading dataset.")

