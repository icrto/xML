## Description
Synthetic Dataset composed by images containing polygons. Each polygon is defined by its **p** and **q** factors, where p is the number of vertices and q represents how these vertices are connected, for example, if q equals 3 then the 1st vertex connects to the 4th and so on. If p equals -1 then the polygon is a circle and the value of q is irrelevant.

Each polygon is randomly placed in the image, considering overlap, occlusion and rotation constraints.

For each generated image, annotations are created in Pascal VOC format, containing information regarding image characteristics and also the classification problem being solved. The polygon one is trying to identify is given, along with information about its existance in the image. If the polygon is present in the image, then bounding boxes are provided for each instance of the desired polygon. As of now, labels for problems like the ones described below are provided:
 * Does polygon X exist or not in the image?
 * How many X polygons exist in the image?
 * Where are polygons X located in the image?

## Configurable Parameters
  * folder='train': directory where dataset is to be stored/imported from
  * config_file=None: configuration file where parameters reside if not None. If None, parameters are passed as constructor args.
  * nr_images=100: number of generated images
  * polygon=[-1, None]: target polygon used to generate annotations given by its parameters p and q (if p equals -1, 
            corresponding to a circle, then q is irrelevant)
  * background_colour=255: background image colour
  * img_height=224: image height
  * img_width=224: image width
  * nr_channels=3: nr of colour channels
  * nr_shapes=20: number os polygons per generated image
  * nr_tries=100: number of tries before algorithm gives up trying to fit polygon inside image
  * rad_min=224/32: minimum possible radius for polygon outer circumference 
  * rad_max=224/16: maximum possible radius for polygon outer circumference 
  * overlap=False: overlap between polygons of the same image if True, no overlap between every two polygons if False
  * occlusion=False: occlusion of polygons on image borders if True, no occlusion if False
  * rotation=True: random rotation of polygons if True, no rotation if False
  * noise=False: add Gaussian noise to image if True
  * min_nr_vertices=3: minimum number of vertices
  * max_nr_vertices=13: maximum number of vertices
  
  If a json config file is given, these parameters are loaded from the file. If not, then parameters are given through arguments of the class constructor.
  
## To Implement
  * annotations for intersection of polygons
  * application of textures to the generated images
  
 ## Examples
 <img src="https://github.com/icrto/xML/blob/master/Synthetic%20Dataset/examples.png" width="1000">
