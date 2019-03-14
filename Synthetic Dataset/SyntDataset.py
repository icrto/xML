import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from lxml import etree as ET
import json

class dataset():

    def __init__(self, config_file=None, folder='train', nr_images=100, polygon=[-1, None], background_colour=255, img_height=224, img_width=224, nr_channels=3,
    nr_shapes=20, nr_tries=100, rad_min=224/32, rad_max=224/16, overlap=False, occlusion=False, rotation=True, noise=False, min_nr_vertices=3, max_nr_vertices=13):

        """ Class constructor
            :param self
            :param folder='train': directory where dataset is to be stored/imported from
            :param config_file=None: configuration file where parameters reside if not None. If None, parameters are passed as constructor args.
            :param nr_images=100: number of generated images
            :param polygon=[-1, None]: target polygon used to generate annotations given by its parameters p and q (if p equals -1, 
            corresponding to a circle, then q is irrelevant)
            :param background_colour=255: background image colour
            :param img_height=224: image height
            :param img_width=224: image width
            :param nr_channels=3: nr of colour channels
            :param nr_shapes=20: number os polygons per generated image
            :param nr_tries=100: number of tries before algorithm gives up trying to fit polygon inside image
            :param rad_min=224/32: minimum possible radius for polygon outer circumference 
            :param rad_max=224/16: maximum possible radius for polygon outer circumference 
            :param overlap=False: overlap between polygons of the same image if True, no overlap between every two polygons if False
            :param occlusion=False: occlusion of polygons on image borders if True, no occlusion if False
            :param rotation=True: random rotation of polygons if True, no rotation if False
            :param noise=False: add Gaussian noise to image if True
            :param min_nr_vertices=3: minimum number of vertices
            :param max_nr_vertices=13: maximum number of vertices

            :return dataset object
        """  


        if(config_file is None):
            self.folder = folder
            self.nr_images = nr_images
            self.polygon = polygon
            self.background_colour = background_colour
            self.img_height = img_height
            self.img_width = img_width
            self.nr_channels = nr_channels
            self.nr_shapes = nr_shapes
            self.nr_tries = nr_tries
            self.rad_min = rad_min
            self.rad_max = rad_max
            self.overlap = overlap
            self.occlusion = occlusion
            self.rotation = rotation
            self.noise = noise
            self.min_nr_vertices = min_nr_vertices
            self.max_nr_vertices = max_nr_vertices
        else:
            if(self.json_parse(config_file) == -1):
                return None

    def create_default_config_file(self):
        """
            :param self

            :return -1 in case of error
        """

        file = {}
        file['folder'] = self.folder
        file['nr_images'] = self.nr_images
        file['polygon'] = []
        file['polygon'].append({
            'p': self.polygon[0],
            'q': self.polygon[1]
        })
        file['background_color'] = self.background_colour
        file['img_height'] = self.img_height
        file['img_width'] = self.img_width
        file['nr_channels'] = self.nr_channels
        file['nr_shapes_per_img'] = self.nr_shapes
        file['nr_tries'] = self.nr_tries
        file['rad_min'] = self.rad_min
        file['rad_max'] = self.rad_max
        file['overlap'] = self.overlap
        file['occlusion'] = self.occlusion
        file['rotation'] = self.rotation
        file['noise'] = self.noise
        file['min_nr_vertices'] = self.min_nr_vertices
        file['max_nr_vertices'] = self.max_nr_vertices

        try:
            with open('config.json', 'w') as outfile:  
                json.dump(file, outfile, indent=4)
        except OSError as err:
            print('Error parsing json file. Please check filename.')
            return -1

    def json_parse(self, file):
        """ json parser
            :param self
            :param file: file to be parsed

            :return -1 in case of error
        """

        try:
            with open(file) as json_file:  
                data = json.load(json_file)
                self.folder = data['folder']
                self.nr_images = data['nr_images']
                self.polygon = np.zeros(2)
                for elem in data['polygon']:
                    self.polygon[0] = elem['p']
                    self.polygon[1] = elem['q']
                self.background_colour = data['background_color'] 
                self.img_height = data['img_height']
                self.img_width = data['img_width']
                self.nr_channels = data['nr_channels']
                self.nr_shapes = data['nr_shapes_per_img']
                self.nr_tries = data['nr_tries']
                self.rad_min = data['rad_min']
                self.rad_max = data['rad_max'] 
                self.overlap = data['overlap']
                self.occlusion = data['occlusion']
                self.rotation = data['rotation']
                self.noise = data['noise']
                self.min_nr_vertices = data['min_nr_vertices']
                self.max_nr_vertices = data['max_nr_vertices']
        except (OSError) as err:
            print('Error parsing json file. Please check filename.')
            return -1

    def create_image(self, name, folder_path):
        """ method to create an image and its xml annotation file in Pascal VOC format
            each image is created according to the class parameters, such as img_height, img_width, nr_channels, background_colour, etc
            each image contains nr_shapes with randomly chosen number of p and q (vertices and jumps) from a set of 
            valid number of vertices which ranges from min_nr_vertices to max_nr_vertices
            each shape's envolving circumference has a randomly chosen radius from min_rad to max_rad
            each shape is randomly positioned in the image considering, or not, overlap, occlusion, rotation and noise
            if nr_tries is exceeded when trying to place a polygon in the image, None is returned
            the generated xml annotation file contains information about the image and the polygon one wants to identify:
                if such polygon exists then a bounding box for each identified polygon is provided
            :param self
            :param name: image name
            :param folder_path: path to where image is stored 
            
            :return created image or None in case of error
        """  

        img = np.ones((self.img_height,self.img_width, self.nr_channels), np.uint8)
        img.fill(self.background_colour)
        centers = []
        count = 0
        
        #xml annotation file    
        ann = ET.Element("annotation")
        ET.SubElement(ann, "folder").text = folder_path
        ET.SubElement(ann, "filename").text = str(name)+'.png'
        size = ET.SubElement(ann, "size")
        ET.SubElement(size, "width").text = str(self.img_width)
        ET.SubElement(size, "height").text = str(self.img_height)
        ET.SubElement(size, "depth").text = str(self.nr_channels)
    
        poly = ET.SubElement(ann, "polygon")
        ET.SubElement(poly, "p").text = str(self.polygon[0])
        ET.SubElement(poly, "q").text = str(self.polygon[1])


        # number of specified polygons found
        polycount = 0

        # number of tries before giving up when a shape cannot be placed in the image
        tries = 0
        while True:
            
            # nr_shapes polygons were drawn...finish
            if(count > self.nr_shapes - 1): break

            # defined number of tries was exceeded...throw an error    
            if(tries > self.nr_tries):
                print('Number of tries exceeded while placing polygon in image.\nPlease specifiy a larger nr_tries or consider lowering nr_shapes per image.')
                return
                

            # only consider this set of valid number of vertices
            valid_interval_vertices = [-1] + list(range(self.min_nr_vertices, self.max_nr_vertices))
            p = np.random.choice(valid_interval_vertices, size=None, replace=True) # choose number of vertices from previous set of values  
            
            if(p != -1):
                q = np.random.randint(1, int(p/2+0.5)) # q < p/2 without loss of generality
                step = 2 * np.pi / p
            else: q = None # p == -1 then the polygon is a circle, so q is theoretically infinite
            
            rad = np.random.randint(self.rad_min, self.rad_max)
            x_orig = np.random.randint(self.img_width)
            y_orig = np.random.randint(self.img_height)

            if(self.occlusion == False):
                if(((self.img_width - 1) - x_orig < rad) or ((x_orig - 0) < rad) or 
                   ((self.img_height - 1) - y_orig < rad) or ((y_orig - 0) < rad)):  # circle will be occluded
                    continue

            overlapping = False
            if(self.overlap == False):
                for aux in centers:
                    if(((x_orig-aux[0])**2 + (y_orig-aux[1])**2) <= (aux[2] + rad)**2): #circles are overlapping
                        overlapping = True
                        break

            # found (x_orig, y_orig) coordinates that respect all imposed restrictions
            if(overlapping == False):
                tries = 0 #reset nr_tries counter because polygon can be placed inside image
                if(self.rotation == True):
                    rot_angle = np.random.uniform(-np.pi, np.pi)
                else:
                    rot_angle = 0
            
                centers.append([x_orig, y_orig, rad])
                count+=1
                vertices = []
                lines = []
                angle = 0.0

                #compute vertices' coordinates
                if(p == -1): #circle
                    img = cv2.circle(img, (x_orig, y_orig), rad, color=(
                        255-self.background_colour), thickness=1, lineType=cv2.LINE_AA)
                else: 
                    for i in range(p):
                        x = x_orig + rad * np.cos(angle)
                        y = y_orig + rad * np.sin(angle)
                        angle += step

                        #rotate around center of polygon
                        x_rot = (x-x_orig)*np.cos(rot_angle) - (y-y_orig)*np.sin(rot_angle) + x_orig
                        y_rot = (x-x_orig)*np.sin(rot_angle) + (y-y_orig)*np.cos(rot_angle) + y_orig

                        vertices.append([x_rot, y_rot])

                    for i in range(0, len(vertices)):
                        lines.append([vertices[i % p], vertices[(i + q) % p]]) 

                    #connect vertices according to q 
                    img = cv2.polylines(img, np.int32(lines), isClosed=True, color=(
                        255-self.background_colour), thickness=1, lineType=cv2.LINE_AA)
                
                #create bounding box in xml file if the polygon that was just drawn equals the desired polygon
                if(p == self.polygon[0] and q == self.polygon[1]):
                    ET.SubElement(ann, "exists").text = str(1)
                    bndbox = ET.SubElement(ann, "bndbox"+str(polycount))
                    ET.SubElement(bndbox, "xmin").text = str(x_orig-rad)
                    ET.SubElement(bndbox, "ymin").text = str(y_orig-rad)
                    ET.SubElement(bndbox, "xmax").text = str(x_orig+rad)
                    ET.SubElement(bndbox, "ymax").text = str(y_orig+rad)
                    polycount += 1

            else: tries += 1 # polygon cannot be placed inside image, trying again

        #no polygons equal to self.polygon were found -> exists = 0
        if(polycount == 0):
            ET.SubElement(ann, "exists").text = str(0)

        try:    
            tree = ET.ElementTree(ann)
            tree.write(folder_path+'/'+str(name)+'.xml', pretty_print=True)
        except:
            print('Error creating xml annotation file.')
            return

        return img


    def noise_gen(self, img, mean, var, noise_type='gaussian'):
        
        """ method for adding noise to an image
            :param self
            :param img: original image
            :param mean: mean of the Gaussian distribution
            :param var: variance of the Gaussian distribution
            :param noise_type='gaussian': type of noise (for now just supports Gaussian noise)

            :return image with noise or None if an error occurred
        """  
        if (noise_type == 'gaussian'):
            row,col,ch = img.shape

            stdev = np.sqrt(var)
            gauss = stdev*np.random.randn(row, col, ch) + mean
            gauss = gauss.reshape(row,col,ch)
            final = img + (255 - self.background_colour) - gauss
            return final.astype('uint8')
        
        return None       

    def create_dataset(self):
        """ method for creating a dataset with nr_images and their respective xml annotation files
            if the destination folder does not yet exist, it is created
            :param self
            
            :return None
        """  
        print("Creating dataset...")
        if((self.polygon[0] != -1) and (self.polygon[1] != None) and (self.polygon[1] >= self.polygon[0] / 2)):
            print("Invalid polygon. Please try again with q < p/2")
            return 
        

        if(not os.path.exists(self.folder)):
            os.makedirs(self.folder)
            
        for i in range(self.nr_images):
            img = self.create_image(i, self.folder)
            if(img is None):
                print('Error creating image.')
                return

            filepath = self.folder+'/'+str(i)+'.png'
            if(self.noise == True):
                img = self.noise_gen(img, 5, 5)
            try:    
                cv2.imwrite(filepath,img)
            except:
                print('Error writing image. Check filepath.')
                return
        
        print("Dataset created.")
        return

    def xml_parse(self, filename):
        """ method for parsing xml annotation files
            :param self
            :param filename: name of xml file to be parsed or None if an error occurred
        
            :return parsed xml file
        """  
        try:
            tree = ET.parse(filename)
        except (OSError, ET.LxmlError)  as err:
            print('Error parsing XML file'+filename)
            return


        root = tree.getroot()
        
        entry = {}
        subelem_pairs = {}
        for elem in root:  
            if(elem.text[0] != '\n'):
                    entry[elem.tag] =  elem.text
            else:
                for subelem in elem:
                    subelem_pairs.setdefault(subelem.tag, [])
                    subelem_pairs[subelem.tag] = subelem.text
                            
                entry[elem.tag] = subelem_pairs
                subelem_pairs = {}
        
        return entry

    def import_dataset(self, folder_path=None, nr_images=None):

        """ method for importing an existing dataset
            :param self
            :param folder_path=None: path of the folder from where the data is to be imported
            :param nr_images=None: number of images to be imported

            :return dictionary containing each image and its annotations or None if an error occurred
        """  
        print("Importing dataset...")

        nr_images = self.nr_images if nr_images is None else nr_images
        folder_path = self.folder if folder_path is None else folder_path

        annotations = {}
        for i in range(nr_images):
            img = cv2.imread(folder_path+'/'+str(i)+'.png')
            if(img is None):
                print('Image could not be loaded. Check folder.')
                return

            aux = self.xml_parse(folder_path+'/'+str(i)+'.xml')
            if(aux is None):
                print('XML file could not be loaded.')
                return

            annotations[i] = aux
            annotations[i]['img'] = img
            
        print("Dataset successfully imported.")
        return annotations



