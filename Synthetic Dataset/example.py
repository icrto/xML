from SyntDataset import dataset
import numpy as np
#example 1 - creating dataset
#created_dataset.create_default_config_file()

#created_dataset = dataset(config_file='config.json')
#created_dataset = dataset()
#if(created_dataset.create_dataset() == -1):
    #print('Error creating dataset')

#else:

#example 2 - importing dataset
imported_dataset = dataset(config_file='config.json')
dtset = imported_dataset.import_dataset()
print(len(dtset))
if(dtset is not None):
    imported_dataset.save("data_simplified.txt", dtset)
    data = imported_dataset.load("data_simplified.txt")
    print(len(data))
    y = [1 if int(datum['exists']) > 0 else 0 for datum in data]
    print(y.count(1))
    print(y.count(0))
    # print(dtset[1])
    # bndboxes = imported_dataset.get_bndboxes(dtset[1])
    # imported_dataset.display_img(dtset[1]['img'], bndboxes)
#else:
   # print('Error loading dataset.')


