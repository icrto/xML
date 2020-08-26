import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from lxml import etree as ET
import json
import pickle


class Dataset:
    def __init__(
        self,
        config_file=None,
        folder="train",
        nr_images=100,
        nr_targets=-1,
        polygon=[-1, None],
        outside_polygon=False,
        background_colour=255,
        img_height=224,
        img_width=224,
        nr_channels=3,
        nr_shapes=20,
        nr_tries=100,
        rad_min=224 / 32,
        rad_max=224 / 16,
        overlap=False,
        occlusion=False,
        rotation=True,
        noise=False,
        min_nr_vertices=3,
        max_nr_vertices=13,
        min_nr_shapes=1,
        max_nr_shapes=20,
        simplified=False,
        no_circles=False,
        poly_colour=False,
        start_index=0,
    ):
        """__init__ [summary]

        Keyword Arguments:
            config_file {str} -- configuration file where parameters reside if not None. If None, parameters are passed as constructor args. (default: {None})
            folder {str} -- folder='train': directory where dataset is to be stored/imported from (default: {"train"})
            nr_images {int} -- number of images to generate (default: {100})
            nr_targets {int} -- number of images to generate with the target polygon present. If -1 then this number is randomly chosen. (default: {-1})
            polygon {list} -- target polygon used to generate annotations given by its parameters p and q (if p equals -1, corresponding to a circle, then q is irrelevant) (default: {[-1, None]})
            outside_polygon {bool} -- polygon to place around target polygon (if simplified mode is set, then this parameter has to be False) (default: {False})
            background_colour {int} -- background image colour (default: {255})
            img_height {int} -- image height (default: {224})
            img_width {int} -- image width (default: {224})
            nr_channels {int} -- number of colour channels (default: {3})
            nr_shapes {int} -- number of polygons per generated image (default: {20})
            nr_tries {int} -- number of tries before the algorithm gives up trying to fit polygon inside image (default: {100})
            rad_min {[type]} -- minimum possible radius for polygon outer circumference (default: {224/32})
            rad_max {[type]} -- maximum possible radius for polygon outer circumference (default: {224/16})
            overlap {bool} -- overlap between polygons of the same image if True, no overlap between every two polygons if False (default: {False})
            occlusion {bool} -- allows occlusion of polygons on image borders if True, no occlusion if False (default: {False})
            rotation {bool} -- defines random rotation of polygons if True, no rotation if False (default: {True})
            noise {bool} -- add Gaussian noise to image if True (default: {False})
            min_nr_vertices {int} -- minimum number of vertices (default: {3})
            max_nr_vertices {int} -- maximum number of vertices (default: {13})
            min_nr_shapes {int} -- minimum number of polygons per image (default: {1})
            max_nr_shapes {int} -- maximum number of polygons per image (default: {20})
            simplified {bool} -- simplified version of the dataset (only triangles and circles) (default: {False})
            no_circles {bool} -- do not draw circles (if simplified mode is set, then this parameter will be ignored) (default: {False})
            poly_colour {bool} -- colour for the target polygon (default: {False})
            start_index {int} -- start index for image naming (useful when one wants to add images to existing dataset) (default: {0})

        """

        if config_file is None:
            self.folder = folder
            self.nr_images = nr_images
            self.nr_targets = nr_targets
            self.polygon = polygon
            self.outside_polygon = outside_polygon
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
            self.min_nr_shapes = min_nr_shapes
            self.max_nr_shapes = max_nr_shapes
            self.simplified = simplified
            self.no_circles = no_circles
            self.poly_colour = poly_colour
            self.start_index = start_index

        else:
            if self.load_json_config_file(config_file) == -1:
                return None

    def create_default_config_file(self):
        """create_default_config_file creates a config json file from the default parameters

        Returns:
            int -- 0 in case of success and -1 if an error occurs
        """

        file = {}
        file["folder"] = self.folder
        file["nr_images"] = self.nr_images
        file["nr_targets"] = self.nr_targets
        file["polygon"] = []
        file["polygon"].append({"p": self.polygon[0], "q": self.polygon[1]})
        if self.outside_polygon is not False:
            file["outside_polygon"] = []
            file["outside_polygon"].append(
                {"p": self.outside_polygon[0], "q": self.outside_polygon[1]}
            )
        else:
            file["outside_polygon"] = None
        file["background_color"] = self.background_colour
        file["img_height"] = self.img_height
        file["img_width"] = self.img_width
        file["nr_channels"] = self.nr_channels
        file["nr_shapes_per_img"] = self.nr_shapes
        file["nr_tries"] = self.nr_tries
        file["rad_min"] = self.rad_min
        file["rad_max"] = self.rad_max
        file["overlap"] = self.overlap
        file["occlusion"] = self.occlusion
        file["rotation"] = self.rotation
        file["noise"] = self.noise
        file["min_nr_vertices"] = self.min_nr_vertices
        file["max_nr_vertices"] = self.max_nr_vertices
        file["min_nr_shapes"] = self.min_nr_shapes
        file["max_nr_shapes"] = self.max_nr_shapes
        file["simplified"] = self.simplified
        file["no_circles"] = self.no_circles
        file["poly_colour"] = self.poly_colour
        file["start_index"] = self.start_index

        try:
            with open("config.json", "w") as outfile:
                json.dump(file, outfile, indent=4)
        except OSError:
            print("Error parsing json file. Please check filename.")
            return -1

        return 0

    def load_json_config_file(self, file):
        """load_json_config_file loads json configuration file

        Arguments:
            file {str} -- json file to parse

        Returns:
            int -- 0 in case of success and -1 if an error occurs
        """

        try:
            with open(file) as json_file:
                data = json.load(json_file)
                self.folder = data["folder"]
                self.nr_images = data["nr_images"]
                self.nr_targets = data["nr_targets"]
                self.polygon = np.zeros(2, dtype=int)
                for elem in data["polygon"]:
                    self.polygon[0] = int(elem["p"])
                    if elem["p"] == -1:
                        self.polygon[1] = 0
                    else:
                        self.polygon[1] = int(elem["q"])
                if data["outside_polygon"] != False:
                    self.outside_polygon = np.zeros(2, dtype=int)
                    for elem in data["outside_polygon"]:
                        self.outside_polygon[0] = int(elem["p"])
                        if elem["p"] == -1:
                            self.outside_polygon[1] = 0
                        else:
                            self.outside_polygon[1] = int(elem["q"])
                else:
                    self.outside_polygon = None
                self.background_colour = data["background_color"]
                self.img_height = data["img_height"]
                self.img_width = data["img_width"]
                self.nr_channels = data["nr_channels"]
                self.nr_shapes = data["nr_shapes_per_img"]
                self.nr_tries = data["nr_tries"]
                self.rad_min = data["rad_min"]
                self.rad_max = data["rad_max"]
                self.overlap = data["overlap"]
                self.occlusion = data["occlusion"]
                self.rotation = data["rotation"]
                self.noise = data["noise"]
                self.min_nr_vertices = data["min_nr_vertices"]
                self.max_nr_vertices = data["max_nr_vertices"]
                self.min_nr_shapes = data["min_nr_shapes"]
                self.max_nr_shapes = data["max_nr_shapes"]
                self.simplified = data["simplified"]
                self.no_circles = data["no_circles"]
                self.poly_colour = data["poly_colour"]
                self.start_index = data["start_index"]

        except (OSError):
            print("Error parsing json file. Please check filename.")
            return -1
        return 0

    def draw(
        self,
        img,
        rotation,
        x_orig,
        y_orig,
        rad,
        p,
        q,
        polygon,
        poly_colour,
        background_colour,
        rot_angle=None,
    ):
        """draw draws the polygons on the provided image

        Arguments:
            img {numpy array} -- image upon which the polygon is drawn
            rotation {bool} -- whether or not to apply rotation to the polygon to be drawn
            x_orig {float} -- polygon's origin in the x axis
            y_orig {float} -- polygon's origin in the y axis
            rad {float} -- radius of the circumference encircling the polygon to be drawn
            p {int} -- number of vertices of the polygon to be drawn
            q {int} -- vertices connection factor
            polygon {list} -- target polygon p and q values
            poly_colour {list} -- RGB values for the colour of the polygon to be drawn
            background_colour {list} -- RGB values for the image background colour

        Keyword Arguments:
            rot_angle {float} -- rotation angle of the polygon to be drawn (default: {None})

        Returns:
            float -- rotation angle defined when no rotation angle was originally given
        """
        if rotation == True:
            if rot_angle is None:
                rot_angle = np.random.uniform(-np.pi, np.pi)
            else:
                rot_angle = rot_angle
        else:
            rot_angle = 0

        vertices = []
        lines = []
        angle = 0.0

        # compute vertices' coordinates
        if p == -1:  # circle
            img = cv2.circle(
                img,
                (x_orig, y_orig),
                rad,
                color=(255 - background_colour),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        else:
            step = 2 * np.pi / p
            for i in range(p):
                x = x_orig + rad * np.cos(angle)
                y = y_orig + rad * np.sin(angle)
                angle += step

                # rotate around center of polygon
                x_rot = (
                    (x - x_orig) * np.cos(rot_angle)
                    - (y - y_orig) * np.sin(rot_angle)
                    + x_orig
                )
                y_rot = (
                    (x - x_orig) * np.sin(rot_angle)
                    + (y - y_orig) * np.cos(rot_angle)
                    + y_orig
                )

                vertices.append([x_rot, y_rot])

            # connect vertices according to q
            for i in range(0, len(vertices)):
                lines.append([vertices[i % p], vertices[(i + q) % p]])

            # if it is the target polygon and a different colour for it is defined
            if (
                poly_colour is not None
                and (p == polygon[0] and q == polygon[1])
                or (p == polygon[0] and polygon[0] == -1)
            ):
                img = cv2.polylines(
                    img,
                    np.int32(lines),
                    isClosed=True,
                    color=(poly_colour),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
            else:
                img = cv2.polylines(
                    img,
                    np.int32(lines),
                    isClosed=True,
                    color=(255 - background_colour),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        return rot_angle

    def create_image(self, name, folder_path):
        """create_image method to create an image and its xml annotation file in Pascal VOC format
            each image is created according to the class parameters, such as img_height, img_width, nr_channels, background_colour, etc
            each image contains nr_shapes (if nr_shapes differs from -1) with randomly chosen number of p and q (vertices and jumps) from a set of valid number of vertices which ranges from min_nr_vertices to max_nr_vertices
            if nr_shapes equals -1 then each image has a randomly defined number of polygons, between min_nr_shapes and max_nr_shapes
            each shape's envolving circumference has a randomly chosen radius from min_rad to max_rad
            each shape is randomly positioned in the image considering, or not, overlap, occlusion, rotation and noise
            if nr_tries is exceeded when trying to place a polygon in the image, nothing is returned
            the generated xml annotation file contains information about the image and the polygon one wants to identify:
                if such polygon exists then a bounding box for each identified polygon is provided

        Arguments:
            name {str} -- image name
            folder_path {str} -- image destination folder

        Returns:
            misc -- returns the created image, the number of polygons it contains and its xml annotation file
        """

        # create the image and fill its background
        img = np.ones((self.img_height, self.img_width, self.nr_channels), np.uint8)
        img.fill(self.background_colour)
        centers = (
            []
        )  # contains the centers of the already drawn polygons (needed to check if the new polygon overlaps existing polygons)
        count = 0
        first = True

        # xml annotation file
        ann = ET.Element("annotation")
        ET.SubElement(ann, "folder").text = folder_path
        ET.SubElement(ann, "filename").text = str(name) + ".png"
        size = ET.SubElement(ann, "size")
        ET.SubElement(size, "width").text = str(self.img_width)
        ET.SubElement(size, "height").text = str(self.img_height)
        ET.SubElement(size, "depth").text = str(self.nr_channels)

        poly = ET.SubElement(ann, "polygon")
        ET.SubElement(poly, "p").text = str(self.polygon[0])
        if self.polygon[0] == -1:
            ET.SubElement(poly, "q").text = "None"
        else:
            ET.SubElement(poly, "q").text = str(self.polygon[1])

        outpoly = ET.SubElement(ann, "outside_polygon")
        if self.outside_polygon is None:
            outpoly.text = "None"
        else:
            ET.SubElement(outpoly, "p").text = str(self.outside_polygon[0])
            if self.outside_polygon[0] == -1:
                ET.SubElement(outpoly, "q").text = "None"
            else:
                ET.SubElement(outpoly, "q").text = str(self.outside_polygon[1])

        # number of specified polygons found
        polycount = 0

        # number of tries before giving up when a polygon cannot be placed in the image
        tries = 0

        # randomly choose the number of shapes per image
        nr_shapes = self.nr_shapes
        if self.nr_shapes == -1:
            valid_nr_shapes = list(range(self.min_nr_shapes, self.max_nr_shapes))
            nr_shapes = np.random.choice(valid_nr_shapes, size=None, replace=True)

        while True:

            # nr_shapes polygons were drawn...finish
            if count > nr_shapes - 1:
                break

            # defined number of tries was exceeded...throw an error
            if tries > self.nr_tries:
                print(
                    "Number of tries exceeded while placing polygon in image.\nPlease specifiy a larger number of tries or consider lowering the number of shapes per image."
                )
                return None, None, None

            if self.simplified:  # only consider one target polygon and circles
                if first:
                    has_target_polygon = np.random.choice([True, False])
                    if has_target_polygon:
                        p = int(self.polygon[0])
                        q = int(self.polygon[1])
                    else:
                        p = -1
                    first = False
                else:
                    p = -1
            else:
                # only consider this set of valid number of vertices
                if self.no_circles:
                    valid_interval_vertices = list(
                        range(self.min_nr_vertices, self.max_nr_vertices)
                    )
                else:
                    valid_interval_vertices = [-1] + list(
                        range(self.min_nr_vertices, self.max_nr_vertices)
                    )

                p = np.random.choice(
                    valid_interval_vertices, size=None, replace=True
                )  # choose number of vertices from previous set of values

            if p != -1:
                q = np.random.randint(
                    1, int(p / 2 + 0.5)
                )  # q < p/2 without loss of generality
            else:
                if self.no_circles:
                    continue  # invalid try (we don't want circles but p equals -1)
                else:
                    q = None  # p == -1 then the polygon is a circle, so q is theoretically infinite

            rad = np.random.randint(self.rad_min, self.rad_max)
            x_orig = np.random.randint(self.img_width)
            y_orig = np.random.randint(self.img_height)

            if self.occlusion == False:
                if (
                    ((self.img_width - 1) - x_orig < rad)
                    or ((x_orig - 0) < rad)
                    or ((self.img_height - 1) - y_orig < rad)
                    or ((y_orig - 0) < rad)
                ):  # circle will be occluded, invalid try
                    continue

            overlapping = False
            if self.overlap == False:
                for aux in centers:  # check every already drawn polygon
                    if ((x_orig - aux[0]) ** 2 + (y_orig - aux[1]) ** 2) <= (
                        aux[2] + rad
                    ) ** 2:  # circles are overlapping
                        overlapping = True
                        break

            # found (x_orig, y_orig) coordinates that respect all imposed restrictions
            if overlapping == False:
                tries = 0  # reset nr_tries counter because polygon can be placed inside image

                centers.append(
                    [x_orig, y_orig, rad]
                )  # add new polygon to list of placed polygon centers
                count += 1  # the image contains one more polygon
                rot_angle = None  # rot_angle will be randomly chosen in the draw method if rotation is True

                # draws the outer polygon when outside_polygon is True; otherwise it is the main (only) polygon to be drawn in this function call
                rot_angle = self.draw(
                    img,
                    self.rotation,
                    x_orig,
                    y_orig,
                    rad,
                    p,
                    q,
                    self.polygon,
                    self.poly_colour,
                    self.background_colour,
                    rot_angle=rot_angle,
                )

                outer_rad = rad
                target = False
                if self.outside_polygon is not None:
                    # if outside_polygon is True then we need to decide if we are going to place a target polygon inside or not
                    if (
                        p == self.outside_polygon[0] and q == self.outside_polygon[1]
                    ) or (
                        p == self.outside_polygon[0] and self.outside_polygon[0] == -1
                    ):
                        encircled = np.random.choice([True, False])
                        if encircled:
                            # draws the target polygon
                            target = True
                            if rad - 5 <= self.rad_min:
                                rad = int(self.rad_min - 5)
                            else:
                                rad = np.random.randint(self.rad_min, rad - 4)
                            centers.append([x_orig, y_orig, rad])
                            self.draw(
                                img,
                                self.rotation,
                                x_orig,
                                y_orig,
                                rad,
                                self.polygon[0],
                                self.polygon[1],
                                self.polygon,
                                self.poly_colour,
                                self.background_colour,
                                rot_angle=rot_angle,
                            )
                        else:  # target polygon includes outside_polygon when outside_polygon is defined, so in this case it is not a positive class instance
                            target = False
                else:
                    if (p == self.polygon[0] and q == self.polygon[1]) or (
                        p == self.polygon[0] and self.polygon[0] == -1
                    ):
                        target = True
                    else:
                        target = False

                # create bounding box in xml file if the polygon that was just drawn equals the desired polygon (plus outside_polygon if True)
                if target == True:
                    bndbox = ET.SubElement(ann, "bndbox" + str(polycount))
                    ET.SubElement(bndbox, "xmin").text = str(x_orig - outer_rad)
                    ET.SubElement(bndbox, "ymin").text = str(y_orig - outer_rad)
                    ET.SubElement(bndbox, "xmax").text = str(x_orig + outer_rad)
                    ET.SubElement(bndbox, "ymax").text = str(y_orig + outer_rad)
                    polycount += 1

            else:
                tries += 1  # polygon cannot be placed inside image, trying again

        # how many target polygons exist in the image
        ET.SubElement(ann, "exists").text = str(polycount)

        return img, polycount, ann

    def noise_gen(self, img, mean, var):
        """noise_gen method for adding guassian noise to an image

        Arguments:
            img {numpy array} -- original image to be modifier
            mean {float} -- mean of the Gaussian distribution
            var {float} -- variance of the Gaussian distribution

        Returns:
            numpy array -- image with added noise
        """

        row, col, ch = img.shape

        stdev = np.sqrt(var)
        gauss = stdev * np.random.randn(row, col, ch) + mean
        gauss = gauss.reshape(row, col, ch)
        final = img + (255 - self.background_colour) - gauss
        return final.astype("uint8")

    def create_dataset(self):
        """create_dataset method for creating a dataset with nr_images and their respective xml annotation files
        (if the destination folder does not yet exist, it is created)

        Returns:
            int -- 0 or -1 in case of error
        """

        if self.simplified and self.outside_polygon is not None:
            print("Cannot create simplified dataset with outside polygon.")
            return -1

        if (
            (self.polygon[0] != -1)
            and (self.polygon[1] != None)
            and (self.polygon[1] >= self.polygon[0] / 2)
        ):
            print("Invalid polygon. Please try again with q < p/2")
            return -1

        if self.outside_polygon is not None:
            if (
                (self.outside_polygon[0] != -1)
                and (self.outside_polygon[1] != None)
                and (self.outside_polygon[1] >= self.outside_polygon[0] / 2)
            ):
                print("Invalid outside polygon. Please try again with q < p/2")
                return -1

        print("Creating dataset...")

        # create folder if it does not exist yet
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        if self.nr_targets == -1:  # randomly generate all images
            for i in range(self.start_index, self.start_index + self.nr_images):
                img, _, ann = self.create_image(i, self.folder)
                if img is None:
                    print("Error creating image.")
                    return -1

                # save annotation xml file
                try:
                    tree = ET.ElementTree(ann)
                    tree.write(
                        os.path.join(self.folder, str(i) + ".xml"), pretty_print=True
                    )
                except:
                    print("Error creating xml annotation file.")

                # save image
                filepath = os.path.join(self.folder, str(i) + ".png")
                if self.noise == True:
                    img = self.noise_gen(img, 5, 5)
                try:
                    cv2.imwrite(filepath, img)
                except:
                    print("Error writing image. Check filepath.")
                    return -1
        else:
            # generate only images with target polygon first
            target_count = 0
            count = 0
            while target_count < self.nr_targets:
                img, polycount, ann = self.create_image(target_count, self.folder)
                if img is None:
                    print("Error creating image.")
                    return -1

                # ignore images without target polygons
                if polycount <= 0:
                    continue
                else:
                    target_count += 1

                # save annotation xml file
                try:
                    tree = ET.ElementTree(ann)
                    tree.write(
                        os.path.join(self.folder, str(target_count - 1) + ".xml"),
                        pretty_print=True,
                    )
                except:
                    print("Error creating xml annotation file.")

                # save image
                filepath = os.path.join(self.folder, str(target_count - 1) + ".png")
                if self.noise == True:
                    img = self.noise_gen(img, 5, 5)
                try:
                    cv2.imwrite(filepath, img)
                except:
                    print("Error writing image. Check filepath.")
                    return -1

            # generate only images without target polygon
            while count < self.nr_images - self.nr_targets:
                img, polycount, ann = self.create_image(
                    count + target_count, self.folder
                )
                if img is None:
                    print("Error creating image.")
                    return -1

                # ignore images with target polygons
                if polycount > 0:
                    continue
                else:
                    count += 1

                # save annotation xml file
                try:
                    tree = ET.ElementTree(ann)
                    tree.write(
                        os.path.join(
                            self.folder, str(count + target_count - 1) + ".xml"
                        ),
                        pretty_print=True,
                    )
                except:
                    print("Error creating xml annotation file.")

                # save image
                filepath = os.path.join(
                    self.folder, str(count + target_count - 1) + ".png"
                )
                if self.noise == True:
                    img = self.noise_gen(img, 5, 5)
                try:
                    cv2.imwrite(filepath, img)
                except:
                    print("Error writing image. Check filepath.")
                    return -1
        print("Dataset created.")
        return 0

    def xml_parse(self, filename):
        """xml_parse method for parsing xml annotation files

        Arguments:
            filename {str} -- name of xml file to be parsed or None if an error occurred

        Returns:
            file -- parsed xml file
        """

        try:
            tree = ET.parse(filename)
        except (OSError, ET.LxmlError):
            print("Error parsing XML file" + filename)
            return

        root = tree.getroot()

        entry = {}
        subelem_pairs = {}
        for elem in root:
            if elem.text[0] != "\n":
                entry[elem.tag] = elem.text
            else:
                for subelem in elem:
                    subelem_pairs.setdefault(subelem.tag, [])
                    subelem_pairs[subelem.tag] = subelem.text

                entry[elem.tag] = subelem_pairs
                subelem_pairs = {}

        return entry

    def import_dataset(self, folder_path=None, nr_images=None):
        """import_dataset  method for importing an existing dataset

        Keyword Arguments:
            folder_path {str} -- path of the folder from where the data is to be imported (default: {None})
            nr_images {int} -- number of images to be imported (default: {None})

        Returns:
            dict -- dictionary containing each image and its annotations or None if an error occurred
        """

        print("Importing dataset...")

        nr_images = self.nr_images if nr_images is None else nr_images
        folder_path = self.folder if folder_path is None else folder_path

        annotations = []
        for i in range(nr_images):
            img = cv2.imread(os.path.join(folder_path, str(i) + ".png"))
            if img is None:
                print("Image could not be loaded. Check folder.")
                return None

            aux = self.xml_parse(os.path.join(folder_path, str(i) + ".xml"))
            if aux is None:
                print("XML file could not be loaded.")
                return None

            aux["img"] = img
            annotations.append(aux)

        print("Dataset successfully imported.")
        return annotations

    def display_img(self, img, bndboxes=None):
        """display_img method that displays an image and its bounding boxes if bndbox is not None

        Arguments:
            img {numpy array} -- image to be displayed

        Keyword Arguments:
            bndboxes {tuple} --  bounding boxes to be displayed (default: {None})
        """

        for bndbox in bndboxes:
            cv2.rectangle(
                img,
                bndbox[0],
                bndbox[1],
                (0, 255, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        plt.imshow(img)
        plt.show()

    def get_bndboxes(self, dict):
        """get_bndboxes method to get bounding boxes from dictionary entry

        Arguments:
            dict {dict} -- dictionary entry from which bounding boxes are to be extracted

        Returns:
            list -- extracted bounding boxes or None if none exist
        """

        nr_bndboxes = int(dict["exists"])
        bndboxes = []
        for i in range(nr_bndboxes):
            bndboxes.append(
                (
                    (
                        int(dict["bndbox" + str(i)]["xmin"]),
                        int(dict["bndbox" + str(i)]["ymin"]),
                    ),
                    (
                        int(dict["bndbox" + str(i)]["xmax"]),
                        int(dict["bndbox" + str(i)]["ymax"]),
                    ),
                )
            )
        return bndboxes

    def save(self, filename, dtset):
        """save method that saves the dataset into a file

        Arguments:
            filename {str} -- path to where the dataset is to be stored
            dtset {list of dict} -- dataset to be stored

        Returns:
            int -- 0 if successfull or -1 if an error occurs
        """
        print("Saving dataset to " + filename)
        try:
            with open(filename, "wb") as fp:  # Pickling
                pickle.dump(dtset, fp)
        except OSError:
            print("Error opening file. Please check filename.")
            return -1

        print("Dataset sucessfully saved")
        return 0

    def load(self, filename):
        """load method that load the dataset from a file

        Arguments:
            filename {str} -- file from which to load the dataset

        Returns:
            list of dict -- loaded dataset or -1 if an error occurs
        """
        print("Loading dataset from " + filename)
        try:
            with open(filename, "rb") as fp:  # Unpickling
                data = pickle.load(fp)
        except OSError:
            print("Error opening file. Please check filename.")
            return -1

        print("Successfully loaded %d images" % len(data))
        return data
