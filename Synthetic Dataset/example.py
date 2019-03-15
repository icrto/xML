from SyntDataset import dataset
import numpy as np
#example 1 - creating dataset
#created_dataset.create_default_config_file()

created_dataset = dataset(config_file='config.json')
#created_dataset = dataset()
#if(created_dataset.create_dataset() == -1):
    #print('Error creating dataset')

#else:

#example 2 - importing dataset
imported_dataset = dataset()
dtset = imported_dataset.import_dataset()
if(dtset is not None):
    print(dtset[5])
    bndboxes = imported_dataset.get_bndboxes(dtset[5])
    imported_dataset.display_img(dtset[5]['img'], bndboxes)
else:
    print('Error loading dataset.')


