## Description

<p align="justify">
          Synthetic Dataset composed by images containing polygons. Each polygon is defined by its <b>p</b> and <b>q</b> factors, where p is the number of vertices and q represents how these vertices are connected, for example, if q equals 3 then the 1st vertex connects to the 4th and so on. If p equals -1 then the polygon is a circle and the value of q is irrelevant.
</p>

<p align="justify">
Each polygon is randomly placed in the image, considering overlap, occlusion and rotation constraints.
</p>
          
<p align="justify">
For each generated image, annotations are created in Pascal VOC format (see this <a href="https://github.com/icrto/xML/blob/master/Synthetic%20Dataset/example.xml">example</a>), containing information regarding image characteristics and also the classification problem being solved. The polygon one is trying to identify is given, along with information about its existance in the image. If the polygon is present in the image, then bounding boxes are provided for each instance of the desired polygon. As of now, labels for problems like the ones described below are provided:
</p>
          
 * Does polygon X exist or not in the image?
 
 * How many X polygons exist in the image?
 
 * Where are polygons X located in the image?
          
## Configurable Parameters
<ul>
  <li><p align="justify">config_file {str} -- configuration file where parameters reside if not None. If None, parameters are passed as constructor args. (default: {None})</p></li>
<li><p align="justify">folder {str} -- folder='train': directory where dataset is to be stored/imported from (default: {"train"})</p></li>
<li><p align="justify">nr_images {int} -- number of images to generate (default: {100})</p></li>
<li><p align="justify">nr_targets {int} -- number of images to generate with the target polygon present. If -1 then this number is randomly chosen. (default: {-1})</p></li>
<li><p align="justify">polygon {list} -- target polygon used to generate annotations given by its parameters p and q (if p equals -1, corresponding to a circle, then q is irrelevant) (default: {[-1, None]})</p></li>
<li><p align="justify">outside_polygon {bool} -- polygon to place around target polygon (if simplified mode is set, then this parameter has to be False) (default: {False})</p></li>
<li><p align="justify">background_colour {int} -- background image colour (default: {255})</p></li>
<li><p align="justify">img_height {int} -- image height (default: {224})</p></li>
<li><p align="justify">img_width {int} -- image width (default: {224})</p></li>
<li><p align="justify">nr_channels {int} -- number of colour channels (default: {3})</p></li>
<li><p align="justify">nr_shapes {int} -- number of polygons per generated image (default: {20})</p></li>
<li><p align="justify">nr_tries {int} -- number of tries before the algorithm gives up trying to fit polygon inside image (default: {100})</p></li>
<li><p align="justify">rad_min {[type]} -- minimum possible radius for polygon outer circumference (default: {224/32})</p></li>
<li><p align="justify">rad_max {[type]} -- maximum possible radius for polygon outer circumference (default: {224/16})</p></li>
<li><p align="justify">overlap {bool} -- overlap between polygons of the same image if True, no overlap between every two polygons if False (default: {False})</p></li>
<li><p align="justify">occlusion {bool} -- allows occlusion of polygons on image borders if True, no occlusion if False (default: {False})</p></li>
<li><p align="justify">rotation {bool} -- defines random rotation of polygons if True, no rotation if False (default: {True})</p></li>
<li><p align="justify">noise {bool} -- add Gaussian noise to image if True (default: {False})</p></li>
<li><p align="justify">min_nr_vertices {int} -- minimum number of vertices (default: {3})</p></li>
<li><p align="justify">max_nr_vertices {int} -- maximum number of vertices (default: {13})</p></li>
<li><p align="justify">min_nr_shapes {int} -- minimum number of polygons per image (default: {1})</p></li>
<li><p align="justify">max_nr_shapes {int} -- maximum number of polygons per image (default: {20})</p></li>
<li><p align="justify">simplified {bool} -- simplified version of the dataset (only triangles and circles) (default: {False})</p></li>
<li><p align="justify">no_circles {bool} -- do not draw circles (if simplified mode is set, then this parameter will be ignored) (default: {False})</p></li>
<li><p align="justify">poly_colour {bool} -- colour for the target polygon (default: {False})</p></li>
<li><p align="justify">start_index {int} -- start index for image naming (useful when one wants to add images to existing dataset) (default: {0})</p></li>

  
 <p align="justify">
          If a <code>json</code> config file is given, these parameters are loaded from the file like <a href="https://github.com/icrto/xML/blob/master/Synthetic%20Dataset/config.json">this</a>. If not, then parameters are given through arguments of the class constructor. You can also use the method <code>create_default_config_file</code> to create a default configuration json file from the default parameters of the <code>Dataset</code> class constructor.
 </p>
   
 ## Examples
 
 <p align="justify">
          Here you can find some examples of images you can generate.
</p>

 <p align="center">
          <img src="https://github.com/icrto/xML/blob/master/Synthetic%20Dataset/examples.png" width="1000">
 </p>
 
 ## Results
 
<p align="justify">
          Here you can find some of the obtained results.
</p>
  
This is the result of the unsupervised explanation loss with α = 0.99 and β = 0.1.

<p align="center">
          <img src="https://github.com/icrto/xML/blob/master/Synthetic%20Dataset/alpha_099_beta_01_img4.png">
</p>
 
 <p align="justify">
          Here we have the result of the unsupervised explanation loss with α = 0.99 and β = 0.25 on a dataset made of triangles only.
 </p>
 
 <p align="center">
          <img src="https://github.com/icrto/xML/blob/master/Synthetic%20Dataset/alpha_099_beta_025_triangles_only.png">
 </p>
