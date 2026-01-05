#
#   @date:
#       28/12/25
#   @author:
#       Tal Ben Ami, 212525257
#       Koren Maavari, 207987314 
# 
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2025
#
import multiprocessing
import random

import numpy as np
from scipy.ndimage import affine_transform
from scipy.ndimage import rotate as scipy_rotate
from scipy.ndimage import shift as scipy_shift


class Worker(multiprocessing.Process):
    def __init__(self, jobs, results, training_data, batch_size):
        super().__init__()

        """ Initialize Worker and it's members.

        Parameters
        ----------
        jobs: JoinableQueue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
		training_data: 
			A tuple of (training images array, image lables array)
		batch_size:
			workers batch size of images (mini batch size)
        
        You should add parameters if you think you need to.
        """
        self.jobs = jobs
        self.results = results
        self.training_data = training_data
        self.batch_size = batch_size

    @staticmethod
    def rotate(image, angle):
        """
        Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : float
            The angle to rotate the image

        Return
        ------
        A numpy array of same shape
        """
        img = image.reshape(28, 28)
        # rotate, reshape=False keeps the output shape
        rotated = scipy_rotate(
            img, angle, reshape=False, order=1, mode="constant", cval=0.0
        )
        return rotated.flatten()

    @staticmethod
    def shift(image, dx, dy):
        """
        Shift given image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis

        Return
        ------
        A numpy array of same shape
        """
        img = image.reshape(28, 28)
        # shift dx cells left and dy cells up (negative direction)
        shifted = scipy_shift(img, shift=(-dy, -dx), order=1, mode="constant", cval=0.0)
        return shifted.flatten()

    @staticmethod
    def add_noise(image, noise):
        """
        Add noise to the image. For each pixel a value is selected
        uniformly from the range [-noise, noise] and added to it.

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        noise : float
            The maximum amount of noise that can be added to a pixel

        Return
        ------
        A numpy array of same shape
        """
        noise_vec = np.random.uniform(-noise, noise, image.shape)
        img_noised = np.clip(image + noise_vec, 0, 1)
        return img_noised

    @staticmethod
    def skew(image, tilt):
        """
        Skew the image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew parameter

        Return
        ------
        A numpy array of same shape
        """
        img = image.reshape(28, 28)
        # result[i][j] = image[i][j + i*tilt], so input coord is [i, j + i*tilt]
        # affine_transform: output[i,j] = input[matrix @ [i,j] + offset]
        # We need matrix @ [i,j] = [i, j + i*tilt] = [[1,0],[tilt,1]] @ [i,j]
        skewed = affine_transform(
            img,
            matrix=[[1, 0], [tilt, 1]],
            offset=0,
            order=1,
            mode="constant",
            cval=0.0,
        )
        return skewed.flatten()

    def process_image(self, image):
        """
        Apply the image process functions.
        Experiment with the random bounds for the functions to see which produces good accuracies.

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        A numpy array of same shape
        """
        img = image.copy()
        # Apply random rotation between -20 and +20 degrees
        angle = random.uniform(-20, 20)
        img = self.rotate(img, angle)

        # Apply random shift between -2 and +2 pixels in both axes
        dx = random.uniform(-2, 2)
        dy = random.uniform(-2, 2)
        img = self.shift(img, dx, dy)

        # Apply random skew between -0.2 and +0.2
        tilt = random.uniform(-0.2, 0.2)
        img = self.skew(img, tilt)

        # Apply random noise with noise amplitude between 0 and 0.2
        noise_amp = random.uniform(0, 0.2)
        img = self.add_noise(img, noise_amp)

        # Make sure pixel values are clipped between 0 and 1
        img = np.clip(img, 0, 1)
        return img

    def run(self):
        """Process images from the jobs queue and add the result to the result queue.
        Hint: you can either generate (i.e sample randomly from the training data)
        the image batches here OR in ip_network.create_batches
        """
        data, labels = self.training_data
        proc_name = self.name
        while True:
            batch_images = []
            batch_labels = []
            batch_idx = self.jobs.get()
            for _ in range(self.batch_size):
                # Poison pill means shutdown
                idx = random.randint(0, len(data) - 1)

                # Append the original image and label
                batch_images.append(data[idx])
                batch_labels.append(labels[idx])

                # Append the augmented image and the same label
                batch_images.append(self.process_image(data[idx]))
                batch_labels.append(labels[idx])
            # print(f"{proc_name}: finished batch number {batch_idx}")
            self.results.put((np.array(batch_images), np.array(batch_labels)))
            self.jobs.task_done()
