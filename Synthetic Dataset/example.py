from SyntDataset import dataset

#example 1 - creating dataset
#created_dataset.create_default_config_file()

created_dataset = dataset(config_file='config.json')
#created_dataset = dataset()
created_dataset.create_dataset()

#example 2 - importing dataset
imported_dataset = dataset()
dtset = imported_dataset.import_dataset()
if(dtset is not None):
    print(dtset[30])
else:
    print('Error loading dataset.')


