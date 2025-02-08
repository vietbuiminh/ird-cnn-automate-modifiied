from skimage import data, io, filters
from shutil import copyfile  # for making folder
from skimage import exposure
import skimage
from skimage.filters import rank
from skimage.morphology import convex_hull_image, erosion, opening, closing, disk
import pandas as pd
from IPython.display import clear_output, display
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from skimage import data, io, filters, draw, util
import os
from skimage.io import imread  # smaller part from skimage
from tensorflow.keras.models import load_model


class LoadCnnLabelImage:
    """
    A class used to load and process CNN label images.

    Methods
    -------
    generate_mask(image, width=550, train=False)
        Generates a mask for the given image.

    clear_edges(im, side_size)
        Clears the edges of the given image.

    label_image_thresh(im, model, minval, maxval)
        Labels the image using a thresholding method.
    """

    def __init__(self, path):
        """
        Initializes the LoadCnnLabelImage object with the given path.

        Parameters
        ----------
        path : str
            The file path of the image to be loaded.
        """
        self.path = path
        self.im = imread(path)

        # PLEASE EDIT THIS BECAUSE I JUST USE THE BASE MODEL FROM THE REPO TO CREATE THIS OBJECT
        model = load_model('04_63_epoch_8639-43402.h5')
        maxval = 43402
        minval = 8639
        side_size = 128

    def plot_images(self, im):
        """
        Plots the raw image, flattened image, and contrast-enhanced image.

        Parameters
        ----------
        im : ndarray
            The input image to be plotted.
        """
        fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 7))
        ax[0].imshow(im, cmap='Greys_r')
        ax[0].title.set_text("Raw Image")
        ax[1].imshow(self.flatten_image(im), cmap='Greys_r')
        ax[1].title.set_text("Flattened Image")
        ax[2].imshow(self.NEWcreateimage(im, self.path, 550,
                     self.create_filter(im)), cmap='Greys_r')
        ax[2].title.set_text("Contrast Enhanced Image")
        plt.show()

    def flatten_image(self, im):
        """
        Flattens the image by normalizing its pixel values after applying specific preprocessing steps.

        Parameters
        ----------
        im : ndarray
        The input image to be flattened.

        Returns
        -------
        ndarray
        The flattened image.
        """
        finim = self.picprep(im, self.path)
        finim = self.NEWapplyfilter(finim, 550, self.create_filter(im))
        return finim

    def generate_mask(self, image, width=550, train=False):
        """
        Generates a mask for the given image.

        Parameters
        ----------
        image : ndarray
            The input image for which the mask is to be generated.
        width : int, optional
            The width of the mask (default is 550).
        train : bool, optional
            Flag to indicate if the mask is for training (default is False).

        Returns
        -------
        ndarray
            The generated mask.
        """
        mask = image
        if width % 2 != 0:
            width = width - 1
        mask = mask < mask.max() * .9
        mask = opening(mask, disk(5))
        mask = convex_hull_image(mask)
        middle_of_core = []
        for y in range(mask.shape[0]):
            if np.sum(np.where(mask[y, :])) > 0:
                middle_of_core.append(np.mean(np.where(mask[y, :])))
        middle_of_core = np.mean(middle_of_core).astype(int)

        if not train:
            mask2 = np.zeros(mask.shape)
            for y in range(mask.shape[0]):
                if np.sum(mask[y, :]) > 0:
                    left = np.array(middle_of_core - width / 2).astype(int)
                    right = np.array(middle_of_core + width / 2).astype(int)
                    mask2[y, left:right] = 1
            return mask * mask2
        else:
            return mask

    def clear_edges(self, im, side_size):
        """
        Clears the edges of the given image.

        Parameters
        ----------
        im : ndarray
            The input image whose edges are to be cleared.
        side_size : int
            The size of the edge to be cleared.

        Returns
        -------
        ndarray
            The image with cleared edges.
        """
        im = np.copy(im)
        im[:int(side_size / 2), :] = -1
        im[-1 * int(side_size / 2):, :] = -1
        im[:, :int(side_size / 2)] = -1
        im[:, -1 * int(side_size / 2):] = -1
        return im

    def label_image_thresh(self, im, model, minval, maxval):
        """
        Labels the image using a thresholding method.

        Parameters
        ----------
        im : ndarray
            The input image to be labeled.
        model : keras.Model
            The trained model used for labeling.
        minval : float
            The minimum value for normalization.
        maxval : float
            The maximum value for normalization.

        Returns
        -------
        ndarray
            The labeled image.
        """
        memory_buffer = 2000
        im2 = self.clear_edges(np.copy(im), side_size)
        side_size = 50  # Define side_size
        im2 = self.clear_edges(np.copy(im), side_size)
        selem = disk(50)
        img_eq = rank.equalize(im2 / np.max(im2), selem=selem)
        img_eq = self.clear_edges(np.copy(img_eq), side_size)
        xv, yv = np.where(img_eq < 100)
        print(xv.size)
        xv = np.array(xv).ravel()
        yv = np.array(yv).ravel()
        L2 = np.zeros(im.shape) - 1
        for i in range(0, xv.size, memory_buffer):
            clear_output()
            if i + memory_buffer < xv.size:
                x_tiny = xv[i:i + memory_buffer]
                y_tiny = yv[i:i + memory_buffer]
            else:
                x_tiny = xv[i:]
                y_tiny = yv[i:]
            to_label_X = np.array([im[int(x - side_size / 2):int(x + side_size / 2),
                                      int(y - side_size / 2):int(y + side_size / 2)]
                                   for x, y in zip(x_tiny, y_tiny)])
            to_label_X = to_label_X.reshape(-1, side_size, side_size, 1)
            to_label_X = (to_label_X - minval) / (maxval - minval)

            print('Labeling pixels ' + str(i) + ' to '
                  + str(i + memory_buffer) + ' out of ' + str(xv.size) + '.\n')
            L = np.argmax(model.predict(to_label_X, verbose=0), axis=1)

            i = 0
            for x, y in zip(x_tiny, y_tiny):
                L2[x, y] = L[i]
                i += 1
        return L2

    def create_filter(self, im):
        """
        Processes an image to create a filter based on specific criteria.
        Parameters:
        im (numpy.ndarray): The input image to be processed.

        Returns:
        numpy.ndarray: A 1D array representing the filter created from the processed image.
        The function performs the following steps:
        1. Crops the input image to a specific region.
        2. Creates a mask by thresholding the cropped image.
        3. Applies morphological opening to the mask.
        4. Computes the convex hull of the mask.
        5. Multiplies the cropped image by the mask to create a new image.
        6. Converts the new image to a pandas DataFrame and computes the mean along the columns.
        7. Converts the resulting DataFrame to a numpy array to form the filter.
        """
        crop = im[0:im.shape[0], 100:1000]
        mask = crop
        mask = mask < mask.max()*.9
        mask = opening(mask, disk(5))
        mask = convex_hull_image(mask)
        create_mask = mask*1
        new_im = crop*create_mask
        ndf = pd.DataFrame(data=new_im)
        ndf = ndf.mean(axis=0)
        filt = np.asarray(ndf)

        return filt

    def picprep(self, im1, path):
        """
        Prepares the image by applying a whiteout if the path ends with 'RAW.tif'.

        Parameters
        ----------
        im1 : ndarray
            The input image to be prepared.
        path : str
            The file path of the image.

        Returns
        -------
        ndarray
            The prepared image.
        """
        whiteout = path.endswith('RAW.tif')
        if whiteout:
            im1[0:im1.shape[0], 0:100] = 65535
        return im1

    def NEWapplyfilter(self, im1, newx, newfilt):
        """
        Applies a filter to the image based on specific criteria.

        Parameters
        ----------
        im1 : ndarray
            The input image to be filtered.
        newx : int
            The width of the core to be filtered.
        newfilt : ndarray
            The filter to be applied.

        Returns
        -------
        ndarray
            The filtered image.
        """
        mask = im1 < im1.max() * .9
        wherecore = np.sum(mask, axis=1)
        mask[wherecore < 450, :] = False

        mask = opening(mask, disk(5))
        mask = convex_hull_image(mask)

        middle_of_core = []
        for y in range(mask.shape[0]):
            if np.sum(np.where(mask[y, :])) > 0:
                middle_of_core.append(np.mean(np.where(mask[y, :])))
        middle_of_core = np.mean(middle_of_core).astype(int)

        xmin = (middle_of_core - newx).astype(np.int64)
        xmax = (middle_of_core + newx).astype(np.int64)

        left = im1[0:im1.shape[0], 0:xmin].astype('float64')
        right = im1[0:im1.shape[0], xmax:len(im1[0])].astype('float64')
        filtered = im1[0:im1.shape[0], xmin:xmax] - newfilt
        newim = np.concatenate((left, filtered, right), axis=1)

        nnewim = newim - newim.min()
        trynew = nnewim / nnewim.max()
        finalim = (trynew * 65535).astype('uint16')
        imeq = exposure.equalize_adapthist(finalim, clip_limit=0.04)
        imeq = (imeq * 65535).astype('uint16')
        return imeq

    def NEWcreateimage(self, im1, path, newx, newfilt):
        """
        Creates a new image by preparing and applying a filter to the input image.

        Parameters
        ----------
        im1 : ndarray
            The input image to be processed.
        path : str
            The file path of the image.
        newx : int
            The width of the core to be filtered.
        newfilt : ndarray
            The filter to be applied.

        Returns
        -------
        ndarray
            The processed image.
        """
        finim = self.picprep(im1, path)
        finim = self.NEWapplyfilter(finim, newx, newfilt)
        return finim

    def label_and_plot(self):
        finim = self.NEWcreateimage(
            self.im, self.path, 550, self.create_filter(self.im))

        finim = finim * self.generate_mask(finim, width=550, train=False)
        label_test_output = self.label_image_thresh(
            finim, model, minval, maxval)
        im_out = label_test_output
        cmap = colors.ListedColormap(["#FFFFFF00", 'red'])
        bounds = [-1, 0.6, 1.0]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 7))
        ax[0].imshow(im, cmap="Greys_r")
        ax[0].title.set_text("Contrast Enhanced Image")
        ax[1].imshow(im, cmap="Greys_r")
        ax[1].imshow(im_out, cmap=cmap, norm=norm)
        ax[1].title.set_text("IRD Labeled Image")
        fig.tight_layout(w_pad=-25, h_pad=0)


# Example usage:
# path = '382-U1537A-16H-6-A_700-860_SHLF9929711_20190428140650_k60_a1_t400_n20_RAW.tif'
# loader = LoadCnnLabelImage(path)
# loader.label_and_plot()
