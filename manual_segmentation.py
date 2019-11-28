# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
{Description}
"""

__author__ = 'Axel Thevenot'
__version__ = '1.4.0'
__maintainer__ = 'Axel Thevenot'
__email__ = 'axel.thevenot@edu.devinci.fr'


import cv2
import numpy as np
import glob
import os


class ManualSegmentation:
    """Class to manullay segment the dataset"""

    def __init__(self, x_save_dir, y_save_dir, n_class, parameters_save_path=None):
        """
        Init the manual segmentation gui
        :param x_save_dir: directory path of the training images
        :param y_save_dir: directory path of the training targets
        :param n_class: number of class into the targets
        :param parameters_save_path: save to the csv that describe the targets
        """
        # paths of training images
        self._X_paths = sorted(glob.glob(os.path.join(x_save_dir, '*.jpg')), key=lambda k: len(k))
        # directory of the targets
        self._Y_dir = y_save_dir
        # current index of paths's list
        self._n = 930 - 1
        # current channel of the target matrix
        self._channel = 0
        # current X and Y to deal with
        self._X, self._Y = None, None
        # number of class which are contained in the target matrix
        self._n_class = n_class

        # Get the parameters save path and load it
        self._param_path = parameters_save_path
        self._load_parameters()

        # load a training images and its target
        self._load()

        # Init the variables to manually interact with the window
        # reference point memory list
        self._ref_p = []
        # right mouse button
        self._r_dragged = False
        # left mouse button
        self._l_dragged = False
        # middle mouse button
        self._m_dragged = False
        # brush size to erase manually
        self._brush_size = 20
        # current mouse position
        self._mouse_pos = [-1, -1]
        # Switch between zoom
        self._zoom_factor = 3

    def _load_parameters(self):
        """
        Load the parameters of the targets if the path is given
        """
        if self._param_path is not None and os.path.exists(self._param_path):
            # get parameters of the targets as a str matrix
            params = np.genfromtxt(self._param_path, delimiter=',', dtype=np.str, comments='---')
            # convert to a dictionary
            params = {column[0]: np.array(column[1:]) for _, column in enumerate(zip(*params))}
            # load the type into an integer
            self._types = np.array([int(value) for value in params['type']])
            # load the hexadecimal BGR color
            self._colors = params['bgr_color']
            # load the name of the different classes
            self._classes = params['class']
        else:
            # arbitrarily set the types to 0
            self._types = np.zeros(self._n_class)
            # randomly choose the colors
            hexadecimal_characters = np.array(['2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'])
            self._colors = ['#' + ''.join(np.random.choice(hexadecimal_characters, 6)) for _ in range(self._n_class)]
            # arbitrarily set the in-order names
            self._classes = ['Class ' + str(i) for i in range(self._n_class)]

    def _click_event(self, event, x, y, flags, param):
        """
        Able to the user to manually interact with the images and their target
        :param event: event raised from the mouse
        :param x: x coordinate of the mouse at the event time
        :param y: y coordinate of the mouse at the event time
        :param flags: flags of the event
        :param param: param of the event
        """
        # may be useful
        if param is not None:
            pass
        # if the right button is pressed we are able to erase the current channel in the target with a square brush
        if event == cv2.EVENT_RBUTTONDOWN:
            self._r_dragged = True
            # get the true x coordinate
            x = x % self._X.shape[1]
            # erase the current target channel according to the brush position (mouse)
            for dx in range(-self._brush_size, self._brush_size + 1):
                for dy in range(-self._brush_size, self._brush_size + 1):
                    if 0 <= x + dx < self._Y.shape[1] and 0 <= y + dy < self._Y.shape[0]:
                        self._Y[y + dy, x + dx, self._channel] = 0

        # if the right button is released, we can't erase anymore
        if event == cv2.EVENT_RBUTTONUP:
            self._r_dragged = False

        # if the middle button is pressed, the scroll of training images/targets is activated
        # else the scroll of target channel is activated
        if event == cv2.EVENT_MBUTTONDOWN:
            self._m_dragged = True
        if event == cv2.EVENT_MBUTTONUP:
            self._m_dragged = False

        # mouse wheel event to navigate between the channels and the samples according to if it is pressed or not
        if event == cv2.EVENT_MOUSEWHEEL:
            if self._m_dragged:
                self._update_image(2 * (flags > 0) - 1)
            else:
                self._update_channel(2 * (flags > 0) - 1)

        # left button released
        if event == cv2.EVENT_LBUTTONUP:
            self._l_dragged = False

        # mouse move
        if event == cv2.EVENT_MOUSEMOVE:
            # update mouse position
            self._mouse_pos = [x, y]
            # activate the current pixel if the left button is pressed and the channel is a pixel by pixel draw type
            if self._l_dragged and self._types[self._channel] == 1:
                self._Y[y, x, self._channel] = 1
            # erase the activated pixels in the channel around the brush if the right button is pressed
            if self._r_dragged:
                x = x % self._X.shape[1]
                for dx in range(-self._brush_size, self._brush_size + 1):
                    for dy in range(-self._brush_size, self._brush_size + 1):
                        if 0 <= x + dx < self._Y.shape[1] and 0 <= y + dy < self._Y.shape[0]:
                            self._Y[y + dy, x + dx, self._channel] = 0

        # left button pressed
        if event == cv2.EVENT_LBUTTONDOWN:
            self._l_dragged = True
            # check if the click is inside the image
            if 0 <= x < self._X.shape[1] and 0 <= y < self._X.shape[0]:
                # check if it is a pixel by pixel draw type
                if self._types[self._channel] == 1:
                    self._Y[y, x, self._channel] = 1
                else:
                    # append the current point to the history
                    self._ref_p.append([x, y])
                    # check if the channel is not a potential unlimited contour type
                    if self._types[self._channel]:
                        # check if the channel is a circle draw type to draw it if the two points are given
                        if self._types[self._channel] == len(self._ref_p) == 2:
                            self._draw_circle()
                        # else wait to reach the number of references points given by the type of the channel
                        elif self._types[self._channel] == len(self._ref_p):
                            self._draw_poly()

    def _draw_poly(self):
        """
        Draw a polynomial from the points given its contours
        """
        # create a new mask
        mask = np.zeros((*self._Y.shape[:2], 3))
        # get the contours points
        pts = np.array([[pt] for pt in self._ref_p])
        # fill the contours in the mask as bitwise then get one channel
        cv2.fillPoly(mask, pts=[pts], color=(1, 1, 1))
        mask = mask[:, :, 0]
        # update the segmented target
        self._Y[:, :, self._channel] = np.array([[1 * (value or mask[i, j])
                                                  for j, value in enumerate(l)]
                                                 for i, l in enumerate(self._Y[:, :, self._channel])])
        # remove the points
        self._ref_p = []

    def _draw_circle(self):
        """
        Draw a circles with the two reference points in memory considering they form a diameter
        """
        # get the two references points in memory
        p1, p2 = self._ref_p
        # get the center of the circle and its radius according to the two bordered selected points
        cx = (p1[0] + p2[0]) // 2
        cy = (p1[1] + p2[1]) // 2
        r = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5 / 2
        # draw the circle
        self._Y[:, :, self._channel] = np.array([[1 * (value or ((j - cx) ** 2 + (i - cy) ** 2 < r ** 2))
                                                  for j, value in enumerate(line)]
                                                 for i, line in enumerate(self._Y[:, :, self._channel])])
        # remove the points
        self._ref_p = []

    def _save(self):
        """
        Save the current target matrix
        """
        name = self._X_paths[self._n].split('\\')[-1].split('.')[0]
        np.save(os.path.join(self._Y_dir, name + '.npy'), self._Y.astype(np.bool))

    def _delete(self):
        """
        Delete the current image and its target
        """
        path_to_remove = self._X_paths.pop(self._n)
        os.remove(path_to_remove)
        # get the target's name
        name = path_to_remove.split('\\')[-1].split('.')[0]
        # get its according target path if it exists
        y_path = glob.glob(os.path.join(self._Y_dir, name + '.npy'))
        # it the target exists, delete it
        if len(y_path):
            os.remove(y_path[0])
        # actualize the current image index
        self._n = self._n % len(self._X_paths)
        # load the new current training sample
        self._load()

    def _load(self):
        """
        Load the current image and its target
        """
        # get the image
        self._X = cv2.imread(self._X_paths[self._n])
        # get its name
        name = self._X_paths[self._n].split('\\')[-1].split('.')[0]
        # get its according target path if it exists
        y_path = glob.glob(os.path.join(self._Y_dir, name + '.npy'))
        # it the target is already known, load it, else set it to an empty matrix
        if len(y_path):
            self._Y = np.load(y_path[0]).astype(np.bool)
        else:
            self._Y = np.zeros((*self._X.shape[:2], self._n_class)).astype(np.bool)

    def _update_image(self, increment=0):
        """
        Update the current image
        :param increment: direction to update
        """
        # save the previous one and its target
        self._save()
        # update the index of the image
        self._n = (self._n + increment) % len(self._X_paths)
        # load the new current image
        self._load()
        # remove the potential references points
        self._ref_p = []

    def _update_channel(self, increment=0):
        """
        Update the current channel
        :param increment: direction to update
        """
        # update the index of the current channel
        self._channel = (self._channel + increment) % self._n_class
        # remove the potential references points
        self._ref_p = []

    @staticmethod
    def _hex2tuple(value, normalize=False):
        """
        Convert an hexadecimal color to its associated tuple
        :param value: value of the hexadecimal color
        :param normalize: normalize the color is needed
        :return: tuple color
        """
        # get the hexadecimal value
        value = value.lstrip('#')
        # get it length
        lv = len(value)
        # get the associated color in 0 to 255 base
        color = np.array([int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)])
        # normalize it if needed
        if normalize:
            color = tuple(color / 255)
        return color

    def _get_x_image(self):
        """
        Get the normalized training image
        :return: normalized image
        """
        return (self._X - np.min(self._X)) / (np.max(self._X) - np.min(self._X))

    def _get_y_image(self):
        """
        Get the colorized target as an image
        :return: colorized target
        """
        # create a black background image
        y_image = np.zeros((*self._Y.shape[:2], 3))
        # for each channel, set the color to the image according to its mask
        for i, color in enumerate(self._colors):
            # get the the mask
            mask = self._Y[:, :, i] > 0
            # update the colorized image
            y_image[np.where(mask)] = self._hex2tuple(color, normalize=True)
        return y_image

    def _get_frame(self):
        """
        Get the gui frame
        :return: gui frame
        """
        # create the image to display the current parameters
        params_img = np.ones((self._X.shape[0] // 10, self._X.shape[1] * 2, 3))

        # set the training image name to the left side of the parameters image

        # get its name
        name = self._X_paths[self._n].split('\\')[-1]
        # compute its height to center the text of the name and compute is fontsize
        text = '{0} - ({1}/{2})'.format(name, self._n + 1, len(self._X_paths))
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_height = size[0][1]
        fontsize = (params_img.shape[0] / text_height) // 2
        # put the name of the image at left
        cv2.putText(params_img,  text, (params_img.shape[0] // 2, params_img.shape[0] * 2 // 3),
                    cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 0, 0), 2)

        # set the current color's class to deal with at right

        # get the up left corner of the color legend rectangle
        up_left = (params_img.shape[1] * 17 // 20, params_img.shape[0] * 1 // 4)
        # get the down right corner of the color legend rectangle
        down_right = (params_img.shape[1] * 19 // 20, params_img.shape[0] * 3 // 4)
        # get the associated color to the current channel
        color = self._hex2tuple(self._colors[self._channel], normalize=True)
        # draw the rectangle to legend the current channel
        cv2.rectangle(params_img, up_left, down_right, color, cv2.FILLED)
        cv2.rectangle(params_img, up_left, down_right, (0, 0, 0), 2)

        # set the current name's class to deal with at the left of its color

        # get the name of the class
        legend = self._classes[self._channel]
        # compute its size to put it a the left of the colored rectangle
        size = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, fontsize, 2)
        # put the text
        cv2.putText(params_img,  legend, (params_img.shape[1] * 16 // 20 - size[0][0], params_img.shape[0] * 2 // 3),
                    cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 0, 0), 2)

        # get the current training image
        x_img = self._get_x_image()
        # get the colorized training target matrix
        y_img = self._get_y_image()
        # apply a filter to see th training image behind the colorized training target matrix
        alpha = 0.3
        y_img = cv2.addWeighted(x_img, alpha, y_img, 1 - alpha, 0)

        # draw each reference points
        for pt in self._ref_p:
            cv2.circle(x_img, tuple(pt), 2, color, cv2.FILLED)

        # draw the brush if the right mouse button is pressed
        if self._r_dragged:
            # get the mouse position
            mouse_x, mouse_y = self._mouse_pos
            # update the current mouse position if it is on the target image
            mouse_x = mouse_x % x_img.shape[1]
            # set the two extremes border points of the brush
            pt1 = (mouse_x - self._brush_size + 3, mouse_y - self._brush_size + 3)
            pt2 = (mouse_x + self._brush_size - 3, mouse_y + self._brush_size - 3)
            # draw it according to the current channel
            cv2.rectangle(x_img, pt1, pt2, (0, 0, 0), 6)
            cv2.rectangle(y_img, pt1, pt2, (0, 0, 0), 6)
            cv2.rectangle(x_img, pt1, pt2, color, 4)
            cv2.rectangle(y_img, pt1, pt2, color, 4)
            cv2.rectangle(x_img, pt1, pt2, (0, 0, 0), 2)
            cv2.rectangle(y_img, pt1, pt2, (0, 0, 0), 2)

        # if the zoom is activated
        if self._zoom_factor > 1:
            # get the current mouse position
            x_mouse, y_mouse = self._mouse_pos
            # check the mouse is inside the image
            if 0 <= x_mouse < x_img.shape[1] and 0 <= y_mouse < x_img.shape[0]:

                # set a window to be zoomed according to a size
                size = int(np.array(x_img.shape[1]) / self._zoom_factor / 10)
                # set the rescaled size of the window according to the zoom factor
                zoom_size = int(size * self._zoom_factor * 2)
                # create a zero-like image to get the window to be zoom
                zoom_img = np.zeros((2 * size, 2 * size, 3))
                # set the pixel to the zoom image if it exists
                for dx in range(-size, size):
                    for dy in range(-size, size):
                        if 0 <= x_mouse + dx < x_img.shape[1] and 0 <= y_mouse + dy < x_img.shape[0]:
                            zoom_img[size + dy, size + dx] = x_img[y_mouse + dy, x_mouse + dx]

                # if it go outside the image
                sy_shift = max(0, zoom_size // 2 - y_mouse)
                sx_shift = max(0, zoom_size // 2 - x_mouse)
                ey_shift = max(0, + zoom_size // 2 + x_mouse - x_img.shape[1])
                ex_shift = max(0, + zoom_size // 2 + y_mouse - x_img.shape[0])

                # rescale the window according to the zoom factor
                zoom_img = cv2.resize(zoom_img, (zoom_size, zoom_size))
                # set black border to the zoom image
                cv2.rectangle(zoom_img, (0, 0), (zoom_size, zoom_size), (0, 0, 0), 3)
                # set the zoom image onto the concatenated training image/target
                sx, sy = y_mouse - zoom_size // 2 + sy_shift, x_mouse - zoom_size // 2 + sx_shift
                ex, ey = y_mouse + zoom_size // 2 - ex_shift, x_mouse + zoom_size // 2 - ey_shift

                # Set the zoomed window
                x_img[sx:ex, sy:ey] = zoom_img[sy_shift:ex-sx+sy_shift, sx_shift:ey-sy+sx_shift]

                # indicated in the target image where the zoom takes place to visualize easily
                # get the two extremes corner points of the zoom window
                pt1 = (x_mouse - size, y_mouse - size)
                pt2 = (x_mouse + size, y_mouse + size)
                # display the zoom image on the colorized target image
                cv2.rectangle(y_img, pt1, pt2, (0, 0, 0), 3)
                cv2.rectangle(y_img, pt1, pt2, (.8, .8, .8), 2)
                cv2.rectangle(y_img, pt1, pt2, (0, 0, 0), 1)

        # concatenate the training image and the colorized target image horizontally
        concat_xy = np.hstack((x_img, y_img))
        # concatenate vertically the training image/target and the current parameter bar image
        return np.vstack((concat_xy, params_img))

    def run(self):
        """
        Run the gui until exit it
        """
        while True:
            # Set the window as normal
            cv2.namedWindow('Manual Segmentation', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Manual Segmentation', self._click_event)

            # Set the window in full screen
            cv2.setWindowProperty('Manual Segmentation', 1, 1)
            # Display the current gui frame
            cv2.imshow('Manual Segmentation', self._get_frame())

            # Continuously wait for a pressed key
            key = cv2.waitKeyEx(1)
            # activate/deactivate the zoom in the gui and its factor
            if key == ord("z"):
                # Switch between zoom factors
                self._zoom_factor = 1 + self._zoom_factor % 5
            # if the enter key is pressed for a potential unlimited contour target, draw it from references points
            if key == 13 and not self._types[self._channel] and len(self._ref_p):
                self._draw_poly()
            # remove the last references points by pressing the return key
            elif key == 8:
                if len(self._ref_p):
                    self._ref_p = self._ref_p[:-1]
                else:
                    self._ref_p = []
            # go to the next image if the right or up arrow is pressed
            elif key == 2555904 or key == 2490368:
                self._update_image(1)
            # go to the previous image if the left or down arrow is pressed
            elif key == 2424832 or key == 2621440:
                self._update_image(-1)
            # delete the current image and its target by
            elif key == 3014656:
                import time
                # avoid to delete two or more training samples at each time
                time.sleep(0.2)
                self._delete()
            # exit the gui if the 'q' key is pressed
            elif key == ord("q"):
                # think to save before to quit
                self._save()
                break

    def reorder(self, x_dir_new, y_dir_new):
        c = 1
        for n in range(len(self._X_paths)):
            # get the image
            X = cv2.imread(self._X_paths[n])
            # get its name
            name = self._X_paths[n].split('\\')[-1].split('.')[0]
            # get its according target path if it exists
            y_path = glob.glob(os.path.join(self._Y_dir, name + '.npy'))
            # it the target is already known, load it, else set it to an empty matrix
            print(name, y_path, c)
            if len(y_path):
                y = np.load(y_path[0]).astype(np.bool)
            else:
                y = np.zeros((*self._X.shape[:2], self._n_class)).astype(np.bool)
            cv2.imwrite(os.path.join(x_dir_new, str(c) + '.jpg'), X)
            np.save(os.path.join(y_dir_new, str(c) + '.npy'), y.astype(np.bool))
            c += 1



if __name__ == '__main__':
    images_save_dir = 'save/train_images'
    targets_save_dir = 'save/train_targets'
    targets_parameters_path = 'save/targets_parameters/targets_parameters.csv'
    ms = ManualSegmentation(images_save_dir, targets_save_dir, 4, parameters_save_path=targets_parameters_path)
    ms.run()



    #ms.reorder('save/train_images2', 'save/train_targets2')
