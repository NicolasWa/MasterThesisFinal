import numpy as np
from skimage.transform import downscale_local_mean, resize
from skimage.color import rgb2hsv, rgb2gray
from skimage.morphology import opening, closing, disk
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import xml.etree.ElementTree as etree
from shapely.geometry import Polygon, MultiPolygon, Point
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

OPENSLIDE_ACTIVE = True
try:
    import openslide
except ImportError:
    OPENSLIDE_ACTIVE = False
    print("WARNING - Openside couldn't be loaded. Some functionalities will not work.")
import tensorflow as tf

def get_tissue_mask_BASIC(fpath, output_size, mag_factor, verbose = True):
    #modif pour generer images
    # modif pour generer images
    #from utils import save_img, save_annot
    #save_folder_path = os.path.join(os.path.dirname(os.getcwd()), "Images mémoire", "Chap 6 Methods")

    disk_size = 10
    if verbose: print("Loading img in low res and converting it to grayscale")
    gray_img = rgb2gray(get_image(fpath, mag=1.25, verbose=False))
    #Modif pour generer images
    #name_grayscale = os.path.basename(fpath) + "_grayscale.png"
    #save_img(gray_img, os.path.join(save_folder_path, name_grayscale))

    if verbose: print("finding the optimal threshold")
    automatic_thres = threshold_otsu(gray_img)
    if verbose: print("Generating a mask based on the optimal threshold")
    tissue_mask = gray_img < automatic_thres
    #modif pour générer images
    #name_otsu_noisy = os.path.basename(fpath) + "_otsu_noisy.png"
    #save_annot(tissue_mask, os.path.join(save_folder_path, name_otsu_noisy))

    if verbose: print("Cleaning the mask")
    tissue_mask_ = opening(closing(tissue_mask, disk(disk_size)), disk(disk_size)) #expensive operation. must be done on downscaled image
    #modif pour générer des images
    # = os.path.basename(fpath) + "_otsu_cleaned.png"
    #save_annot(tissue_mask_, os.path.join(save_folder_path, name_otsu_cleaned))

    if verbose: print("Resizing the mask to the original shape")
    tissue_mask_resized = resize(tissue_mask_, (output_size[0], output_size[1])).astype('uint8')
    return tissue_mask_resized

def get_image(fpath, mag=1.25, verbose=False):
    '''Load a whole-slide image (ndpi, svs) & extract image @ 1.25x or 2.5x magnification'''
    if( OPENSLIDE_ACTIVE == False ):
        print("Error - Openslide could not be loaded")
        return False
    if( verbose ):
        print(f"Loading RGB image @ {mag}x magnification")
    slide = openslide.OpenSlide(fpath)
    op = float(slide.properties['openslide.objective-power']) # maximum magnification
    down_factor = op/mag
    level = slide.get_best_level_for_downsample(down_factor)
    relative_down = down_factor / slide.level_downsamples[level]

    rgb_image = slide.read_region((0,0), level, slide.level_dimensions[level])
    newsize = np.array((slide.level_dimensions[level][0]/relative_down, slide.level_dimensions[level][1]/relative_down)).astype('int')
    rgb_image = np.array(rgb_image.resize(newsize))[:,:,:3].astype('uint8')
    return rgb_image

def get_image_anno_tissue(fpath, apath, mag=1.25, verbose=False):
    '''Load a whole-slide image (ndpi, svs) & extract image & annotation mask @ given magnification'''
    if( OPENSLIDE_ACTIVE == False ):
        print("Error - Openslide could not be loaded")
        return False
    if( verbose ):
        print(f"Loading RGB image @ {mag}x magnification")
    slide = openslide.OpenSlide(fpath)
    op = float(slide.properties['openslide.objective-power']) # maximum magnification
    down_factor = op/mag
    level = slide.get_best_level_for_downsample(down_factor)
    relative_down = down_factor / slide.level_downsamples[level]

    rgb_image = slide.read_region((0,0), level, slide.level_dimensions[level])
    newsize = np.array((slide.level_dimensions[level][0]/relative_down, slide.level_dimensions[level][1]/relative_down)).astype('int')
    rgb_image = np.array(rgb_image.resize(newsize))[:,:,:3].astype('uint8')

    osa = OpenSlideAnnotation(apath, slide)
    anno_mask = (osa.getMask([level])[0]).astype("uint8")
    tissue_mask = get_tissue_mask_BASIC(fpath, anno_mask.shape, mag, verbose = True)
    anno_mask = (anno_mask > 0) * tissue_mask * 255
    return rgb_image, anno_mask, tissue_mask

""" FUNCTION NEEDED (says A. Foucart)"""
def get_image_and_anno(fpath, apath, mag=1.25, verbose=False):
    '''Load a whole-slide image (ndpi, svs) & extract image & annotation mask @ given magnification'''
    if( OPENSLIDE_ACTIVE == False ):
        print("Error - Openslide could not be loaded")
        return False
    if( verbose ):
        print(f"Loading RGB image @ {mag}x magnification")
    slide = openslide.OpenSlide(fpath)
    op = float(slide.properties['openslide.objective-power']) # maximum magnification
    down_factor = op/mag
    level = slide.get_best_level_for_downsample(down_factor)
    relative_down = down_factor / slide.level_downsamples[level]

    rgb_image = slide.read_region((0,0), level, slide.level_dimensions[level])
    newsize = np.array((slide.level_dimensions[level][0]/relative_down, slide.level_dimensions[level][1]/relative_down)).astype('int')
    rgb_image = np.array(rgb_image.resize(newsize))[:,:,:3].astype('uint8')

    osa = OpenSlideAnnotation(apath, slide)
    anno_mask = (osa.getMask([level])[0]).astype("uint8")


    return rgb_image, anno_mask #.astype("uint8")

def imageWithOverlay(img, mask):
    '''Display green on the non-tumorous regions and keeps the tumorous region intact'''
    imask = mask==False
    output_image = img.copy()
    output_image[imask,:] *= np.array([0,0,0]).astype('uint8') #[0,1,0]

    return output_image

def blend2Images(img, mask):
    '''blend2Images mathod From HistoQC (Janowczyk et al, 2019)

    Source: https://github.com/choosehappy/HistoQC
    Produces output image with artefact regions in green.
    '''
    if (img.ndim == 3):
        img = rgb2gray(img)
    if (mask.ndim == 3):
        mask = rgb2gray(mask)
    img = img[:, :, None] * 1.0  # can't use boolean
    mask = mask[:, :, None] * 1.0
    out = np.concatenate((mask, img, mask), 2)
    return out


class OpenSlideAnnotation:
    '''Reads .ndpa annotations files & produce annotation mask'''

    def __init__(self, fname, slide):
        self.fname = fname
        self.slide = slide
        tree = etree.parse(fname)    # Annotations
        self.root = tree.getroot()

        mppx = float(self.slide.properties['openslide.mpp-x']) #um/px
        mppy = float(self.slide.properties['openslide.mpp-y'])
        self.nppx = mppx*1000 # nm/px
        self.nppy = mppy*1000
        self.xoff = float(self.slide.properties['hamamatsu.XOffsetFromSlideCentre']) # in nm
        self.yoff = float(self.slide.properties['hamamatsu.YOffsetFromSlideCentre'])
        self.cx,self.cy = self.slide.level_dimensions[0][0]/2, self.slide.level_dimensions[0][1]/2

        self.C = np.array([self.cx, self.cy])
        self.T = np.array([self.xoff, self.yoff])
        self.S = np.array([self.nppx, self.nppy])

    def getAllAnnotations(self):
        '''Generator to get all the annotations in the XML file.
        Applies the offset & the conversion to get the coordinates in pixels'''
        pointlists = [ann.find('annotation').find('pointlist') for ann in self.root]
        for plist in pointlists:
            points = [[float(p.find('x').text),float(p.find('y').text)] for p in plist.findall('point')]
            points += [points[0]]
            points = self.C+(np.array(points)-self.T)/self.S
            yield points

    def getMask(self, levels):
        '''Get annotation masks at the required levels of magnification.'''
        ratios = [(self.slide.dimensions[0]//self.slide.level_dimensions[level][0], self.slide.dimensions[1]//self.slide.level_dimensions[level][1]) for level in levels]
        masks = [Image.new('L', self.slide.level_dimensions[level], 0) for level in levels]
        for points in self.getAllAnnotations():
            points_ = [list(np.round(points/np.array(r)).astype('int').flatten()) for r in ratios]
            for i in range(len(levels)):
                ImageDraw.Draw(masks[i]).polygon(points_[i], outline=255, fill=255)
        masks = [np.array(mask) for mask in masks]

        return masks

""" Modif:"""

def tiling_WSI_and_annot(img, mask, img_name, mask_name, tile_size_y = 612, tile_size_x = 512 , verbose = True):
    if verbose == True:
        print("entering in the tiling function")
    n_tiles_y = img.shape[0]//tile_size_y
    n_tiles_x = img.shape[1]//tile_size_x
    for i in range (n_tiles_y):
        for j in range (n_tiles_x):
            img_temp = img[i*tile_size_y:(i+1)*tile_size_y, j*tile_size_x: (j+1)*tile_size_x, :]
            mask_temp = mask[i*tile_size_y:(i+1)*tile_size_y, j*tile_size_x: (j+1)*tile_size_x]
            tile_img_name = img_name + "_" + str(i) + "_" + str(j)
            tile_mask_name = mask_name +  "_" + str(i) + "_" + str(j)
            im = Image.fromarray(img_temp)
            im.save("./tiles/" + tile_img_name + ".jpeg")
            im_mask = Image.fromarray(mask_temp)
            im_mask.save("./targets/" + tile_mask_name + ".jpeg")
    return None

""" Will be used to reconstruct the prediction mask """
def reconstruct_mask(img, mask, img_name, mask_name, tile_size_y = 612, tile_size_x = 512 , verbose = True):
    if verbose == True:
        print("entering in the reconstruct function")
    n_tiles_y = img.shape[0]//tile_size_y
    n_tiles_x = img.shape[1]//tile_size_x
    mask_reconstructed = np.zeros(mask.shape)
    for i in range (n_tiles_y):
        for j in range (n_tiles_x):
            tile_mask_name = mask_name +  "_" + str(i) + "_" + str(j)
            im_mask = Image.open("./tiles/" + tile_mask_name + ".jpeg")
            data = np.array(im_mask)
            mask_reconstructed[i*tile_size_y:(i+1)*tile_size_y, j*tile_size_x:(j+1)*tile_size_x] = data

    im_mask_reconstructed = Image.fromarray(mask_reconstructed)
    im_mask_reconstructed = im_mask_reconstructed.convert("L") #needed because it is a grayscale image
    im_mask_reconstructed.save(mask_name + "_reconstructed" + ".jpeg")
    #return mask_reconstructed
    return None
