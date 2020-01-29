# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
User friendly interface to manually segment images
"""

__author__ = 'Axel Thevenot'
__version__ = '2.1.0'
__maintainer__ = 'Axel Thevenot'
__email__ = 'axel.thevenot@edu.devinci.fr'

import os
import glob
import argparse

import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('x_save_dir', help='Train images directory')
parser.add_argument('y_save_dir', help='Target images directory')
parser.add_argument('config_path', help='Path to csv config file')
args = parser.parse_args()


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


keys = {
    'next_channel': ord('e'),           # ASCII number = 101
    'previous_channel': ord('d'),       # ASCII number = 100
    'next_image': ord('f'),             # ASCII number = 102
    'previous_image': ord('s'),         # ASCII number = 115
    'zoom': ord('z'),                   # ASCII number = 122
    'validate': 13,                     # ASCII number = 13
    'undo': ord('u'),                   # ASCII number = 117
    'brush': ord('b'),                  # ASCII number = 98
    'delete': 8,                        # ASCII number = 8
    'quit': ord('q')                    # ASCII number = 113
}

# to transform the dictionary into an object
keys = objectview(keys)



class ManualSegmentation:
    """Class to manullay segment the dataset"""

    def __init__(self, x_save_dir, y_save_dir, config_save_path):
        """
        Init the manual segmentation gui
        :param x_save_dir: directory path of the training images
        :param y_save_dir: directory path of the training targets
        :param config_save_path: save to the csv that describe the targets
        """
        # number of _classes
        self.n_class = None
        # shapes of draw of the different classes
        self.shapes = None
        # hexadecimal BGR color of the different classes
        self.colors = None
        # name of the different classes
        self.class_names = None
        # load the config save path to fill the attributes above
        self.load_config(config_save_path)

        # paths of training images sorted by their ID number
        self.X_paths = glob.glob(os.path.join(x_save_dir, "*.jpg"))
        self.X_paths = sorted(self.X_paths, key=lambda k: (len(k), k))
        # directory of the targets
        self.Y_dir = y_save_dir
        # current index of paths's list
        self.n = 0
        # current channel of the target matrix
        self.channel = 0
        # current X (training) and Y (target) to deal with
        self.X, self.Y = None, None
        # load a training images and its target
        self.load()

        # Init the variables to manually interact with the window
        # reference point memory list
        self.ref_p = []
        # activation of the eraser
        self.brush = False
        # brush size to erase manually
        self.brush_size = 20

        # current mouse position
        self.mouse_pos = [0, 0]
        # left mouse button is pressed
        self.l_pressed = False
        # Switch between zoom
        self.zoom_factor = 3

    def load_config(self, path):
        """
        Load the config of the targets if the path is given
        :param path: path to the config
        """
        if os.path.exists(path):
            # get config of the targets as a str matrix
            config = np.genfromtxt(
                path, delimiter=",", dtype=np.str, comments=None
            )
            # convert to a dictionary
            config = {
                column[0]: np.array(column[1:])
                for _, column in enumerate(zip(*config))
            }
            # load the shapes into an integer
            self.shapes = np.array([int(value) for value in config['shape']])
            # load the hexadecimal BGR color
            self.colors = config['bgr_color']
            # load the name of the different classes
            self.class_names = config['class']
            # Get the number of _classes
            self.n_class = len(self.class_names)
        else:
            print(f"The path {path} does not exist")

    def load(self):
        """
        Load the current image and its target
        """
        # get the input image
        self.X = cv2.imread(self.X_paths[self.n])

        # get the target image
        # get its name removing folders in the X path anoted with `//` or `\`
        file_name = self.X_paths[self.n].split("\\")[-1].split("/")[-1]
        # split by the `.` to remove the extension to get the name
        name = file_name.split(".")[0]
        # get its according target path if it exists
        y_path = glob.glob(os.path.join(self.Y_dir, name + ".npy"))

        # if the target is already existing, load it
        if len(y_path):
            self.Y = np.load(y_path[0])
        # else set it to an empty matrix (full of zeros)
        else:
            self.Y = np.zeros((*self.X.shape[:2], self.n_class))


    def save(self):
        """
        Save the current target matrix
        """
        # get its name by removing folders in the path anoted with `//` or `\`
        file_name = self.X_paths[self.n].split("\\")[-1].split("/")[-1]
        # split by the `.` to remove the extension to get the name
        name = file_name.split(".")[0]
        # convert to boolean to gain space
        self.Y = self.Y.astype(np.bool)
        # save it
        np.save(os.path.join(self.Y_dir, name + ".npy"), self.Y)

    def delete(self):
        """
        Delete the current image and its target
        """
        path_to_remove = self.X_paths.pop(self.n)
        os.remove(path_to_remove)
        # get the target's name
        name = path_to_remove.split("\\")[-1].split("/")[-1].split(".")[0]
        # get its according target path if it exists
        y_path = glob.glob(os.path.join(self.Y_dir, name + ".npy"))
        # it the target exists, delete it
        if len(y_path):
            os.remove(y_path[0])
        # actualize the current image index
        self.n = self.n % len(self.X_paths)
        # load the new current training sample
        self.load()

    def update_image(self, increment=0):
        """
        Update the current image
        :param increment: direction to update
        """
        # save the previous one and its target
        self.save()
        # update the index of the image
        self.n = (self.n + increment) % len(self.X_paths)
        # load the new current image
        self.load()
        # remove the potential references points
        self.ref_p = []

    def update_channel(self, increment=0):
        """
        Update the current channel
        :param increment: direction to update
        """
        # update the index of the current channel
        self.channel = (self.channel + increment) % self.n_class
        # remove the potential references points
        self.ref_p = []

    def click_event(self, event, x, y, flags, param):
        """
        Able to the user to manually interact with the images and their target
        :param event: event raised from the mouse
        :param x: x coordinate of the mouse at the event time
        :param y: y coordinate of the mouse at the event time
        :param flags: flags of the event
        :param param: param of the event
        """

        # mouse move
        if event == cv2.EVENT_MOUSEMOVE:
            # update mouse position
            self.mouse_pos = [x, y]
            # check if the left button is pressed to have an action
            if self.l_pressed:
                # erase the activated pixels in the channel
                if self.brush:
                    self.set_brush_eraser()
                # else draw pixel if  it is `pixel by pixel` draw shape
                elif self.shapes[self.channel] == 1:
                    self.set_pixel()

        if event == cv2.EVENT_LBUTTONUP:
            self.l_pressed = False

        # left button pressed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.l_pressed = True
            # check if the brush eraser is activated
            if self.brush:
                self.set_brush_eraser()
            # check if the click is inside the image
            elif 0 <= x < self.X.shape[1] and 0 <= y < self.X.shape[0]:
                # check if it is a pixel by pixel draw shape
                if self.shapes[self.channel] == 1:
                    self.set_pixel()
                else:
                    # append the current point to the history
                    self.ref_p.append([x, y])
                    # check if the channel is an unlimited contour shape
                    if self.shapes[self.channel]:
                        # check if the channel is a circle draw shape
                        # and if the two points are given
                        if self.shapes[self.channel] == len(self.ref_p) == 2:
                            self.set_circle()
                        # else wait to reach the number of references points
                        # given by the shape of the channel
                        elif self.shapes[self.channel] == len(self.ref_p):
                            self.set_poly()

    def set_pixel(self):
        """
        Set value to 1 at the mouse position in the case of `1` draw shape
        """
        # draw the pixel according at the mouse position
        x, y = self.mouse_pos
        self.Y[y, x, self.channel] = 1


    def set_brush_eraser(self):
        """
        Erase the target image
        according to the brush size and the mouse position
        """
        # get the mouse position on the image
        x, y = self.mouse_pos
        x = x % self.X.shape[1]
        # erase by shifting index and set the pixels to zero
        for dx in range(-self.brush_size, self.brush_size + 1):
            for dy in range(-self.brush_size, self.brush_size + 1):
                # continue if the point is not in the image
                if not 0 <= x + dx < self.Y.shape[1]:
                    continue
                if not 0 <= y + dy < self.Y.shape[0]:
                    continue
                self.Y[y + dy, x + dx, self.channel] = 0

    def set_poly(self):
        """
        Draw a polynomial from the points given its contours
        """
        if len(self.ref_p) > 1:
            # create a new mask
            mask = np.zeros((*self.Y.shape[:2], 3))
            # get the contours points
            pts = np.array([[pt] for pt in self.ref_p])
            # fill the contours in the mask as bitwise then get one channel
            cv2.fillPoly(mask, pts=[pts], color=(1, 1, 1))
            mask = mask[:, :, 0]
            mask = [[1 * (value or mask[i, j])
                        for j, value in enumerate(l)]
                            for i, l in enumerate(self.Y[:, :, self.channel])]
            # update the segmented target
            self.Y[:, :, self.channel] = np.array(mask)
            # remove the points
            self.ref_p = []

    def set_circle(self):
        """
        Draw a circles with the two reference points in memory
        considering they give us the diameter of the circle
        """
        # get the two references points in memory
        p1, p2 = self.ref_p
        # get the center of the circle and its radius
        # according to the two bordered selected points
        cx = (p1[0] + p2[0]) // 2
        cy = (p1[1] + p2[1]) // 2
        r = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5 / 2
        # get the mask of the circle
        mask = [[1 * (value or ((j - cx) ** 2 + (i - cy) ** 2 < r ** 2))
                    for j, value in enumerate(line)]
                        for i, line in enumerate(self.Y[:, :, self.channel])]
        # draw the circle
        self.Y[:, :, self.channel] = np.array(mask)
        # remove the points
        self.ref_p = []

    def draw_circle_visualization(self, x_img, y_img, color):
        """
        Draw the circle in the case of `2` draw typ to visiualize the circle
        before to set the 2nd and last refernce point
        :param x_img: input image
        :param x_img: colorized target image
        :param color: color of the current channel
        """
        p1, p2 = self.ref_p[0], self.mouse_pos
        # get the center of the circle and its radius
        # according to the two bordered selected points
        cx = (p1[0] + p2[0]) // 2
        cy = (p1[1] + p2[1]) // 2
        r = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5 / 2
        # draw the circle to visualize the rendered segmented circle
        cv2.circle(x_img, (cx, cy), int(r), color, 1)
        cv2.circle(y_img, (cx, cy), int(r), color, 1)

    @staticmethod
    def hex2tuple(value, normalize=False):
        """
        Convert an hexadecimal color to its associated tuple
        :param value: value of the hexadecimal color
        :param normalize: normalize the color is needed
        :return: tuple color
        """
        # get the hexadecimal value
        value = value.lstrip("#")
        # get it length
        lv = len(value)
        # get the associated color in 0 to 255 base
        color = np.array([int(value[i : i + lv // 3], 16)
                            for i in range(0, lv, lv // 3)])
        # normalize it if needed
        if normalize:
            color = tuple(color / 255)
        return color

    def get_x_image(self):
        """
        Get the normalized training image
        :return: normalized image
        """
        return (self.X - np.min(self.X)) / (np.max(self.X) - np.min(self.X))

    def get_y_image(self):
        """
        Get the colorized target as an image
        :return: colorized target
        """
        # create a black background image
        y_image = np.zeros((*self.Y.shape[:2], 3))
        # for each channel, set the color to the image according to its mask
        for i, color in enumerate(self.colors):
            # get the the mask
            mask = self.Y[:, :, i] > 0
            # update the colorized image
            y_image[np.where(mask)] = self.hex2tuple(color, normalize=True)
        return y_image

    def get_params_image(self):
        """
        Build the parameters bar to display information
        at the bottom of the gui :
            - The name of the current image at left
            - The name and color of the current channel at right
        :return: normalized image
        """
        # create the image to display the current parameters
        params_img = np.ones((self.X.shape[0] // 10, self.X.shape[1] * 2, 3))

        # get its name
        name = self.X_paths[self.n].split("\\")[-1].split("/")[-1]
        # compute its height to center the text of the name and compute is fontsize
        text = "{0} - ({1}/{2})".format(name, self.n + 1, len(self.X_paths))
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_height = size[0][1]
        fontsize = (params_img.shape[0] / text_height) // 2
        # put the name of the image at left
        cv2.putText(
            params_img,
            text,
            (params_img.shape[0] // 2, params_img.shape[0] * 2 // 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontsize,
            (0, 0, 0),
            2,
        )

        # set the current color's class to deal with at right

        # get the up left corner of the color legend rectangle
        up_left = (
            params_img.shape[1] * 17 // 20,
            params_img.shape[0] * 1 // 4,
        )
        # get the down right corner of the color legend rectangle
        down_right = (
            params_img.shape[1] * 19 // 20,
            params_img.shape[0] * 3 // 4,
        )
        # draw the rectangle to legend the current channel
        color = self.hex2tuple(self.colors[self.channel], normalize=True)
        cv2.rectangle(params_img, up_left, down_right, color, cv2.FILLED)
        cv2.rectangle(params_img, up_left, down_right, (0, 0, 0), 2)

        # set the current name's class to deal with at the left of its color

        # get the name of the class
        legend = self.class_names[self.channel]
        # compute its size to put it a the left of the colored rectangle
        size = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, fontsize, 2)
        # put the text
        cv2.putText(
            params_img,
            legend,
            (
                params_img.shape[1] * 16 // 20 - size[0][0],
                params_img.shape[0] * 2 // 3,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontsize,
            (0, 0, 0),
            2,
        )
        return params_img

    def draw_brush(self, x_img, y_img, color):
        """
        Draw the brush if activated on the images
        :param x_img: input image
        :param x_img: colorized target image
        :param color: color of the current channel
        """
        # get the mouse position
        mouse_x, mouse_y = self.mouse_pos
        # update the current mouse position if it is on the target image
        mouse_x = mouse_x % x_img.shape[1]
        # set the two extremes border points of the brush
        pt1 = (mouse_x - self.brush_size + 3, mouse_y - self.brush_size + 3)
        pt2 = (mouse_x + self.brush_size - 3, mouse_y + self.brush_size - 3)
        # draw it according to the current channel
        cv2.rectangle(x_img, pt1, pt2, (0, 0, 0), 6)
        cv2.rectangle(y_img, pt1, pt2, (0, 0, 0), 6)
        cv2.rectangle(x_img, pt1, pt2, color, 4)
        cv2.rectangle(y_img, pt1, pt2, color, 4)
        cv2.rectangle(x_img, pt1, pt2, (0, 0, 0), 2)
        cv2.rectangle(y_img, pt1, pt2, (0, 0, 0), 2)

    def draw_zoom_window(self, x_img, y_img):
        """
        Draw the zoom window if activated on the input image
        and draw its position on the target image
        :param x_img: input image
        :param x_img: colorized target image
        """
        # get the current mouse position
        x, y = self.mouse_pos
        # check the mouse is inside the image
        if 0 <= x < x_img.shape[1] and 0 <= y < x_img.shape[0]:
            # set a window to be zoomed according to a size
            size = int(np.array(x_img.shape[1]) / self.zoom_factor / 10)
            # set the rescaled size of the window according to the zoom factor
            zoom_size = int(size * self.zoom_factor * 2)
            # create a zero-like image to get the window to be zoom
            zoom_img = np.zeros((2 * size, 2 * size, 3))
            # set the pixel of the zoom image
            for dx in range(-size, size):
                for dy in range(-size, size):
                    # continue if the point is not in the image
                    if not 0 <= x + dx < x_img.shape[1]:
                        continue
                    if not 0 <= y + dy < x_img.shape[0]:
                        continue
                    zoom_img[size + dy, size + dx] = x_img[y + dy, x + dx]
            # if the zoom window is shifting outside the image
            sy_shift = max(0, zoom_size // 2 - y)
            sx_shift = max(0, zoom_size // 2 - x)
            ey_shift = max(0, +zoom_size // 2 + x - x_img.shape[1])
            ex_shift = max(0, +zoom_size // 2 + y - x_img.shape[0])

            # rescale the window according to the zoom factor
            zoom_img = cv2.resize(zoom_img, (zoom_size, zoom_size))
            # set black border to the zoom image
            cv2.rectangle(
                zoom_img, (0, 0), (zoom_size, zoom_size), (0, 0, 0), 3
            )
            # set the zoom image onto the concatenated training image/target
            sx, sy = (
                y - zoom_size // 2 + sy_shift,
                x - zoom_size // 2 + sx_shift,
            )
            ex, ey = (
                y + zoom_size // 2 - ex_shift,
                x + zoom_size // 2 - ey_shift,
            )
            # superimpose the zoomed window to the input image
            x_img[sx:ex, sy:ey] = zoom_img[
                sy_shift:ex - sx + sy_shift, sx_shift:ey - sy + sx_shift
            ]

            # draw in the target image where the zoom takes placey
            # get the two extremes corner points of the zoom window
            pt1 = (x - size, y - size)
            pt2 = (x + size, y + size)
            # display the zoom image on the colorized target image
            cv2.rectangle(y_img, pt1, pt2, (0, 0, 0), 3)
            cv2.rectangle(y_img, pt1, pt2, (0.8, 0.8, 0.8), 2)
            cv2.rectangle(y_img, pt1, pt2, (0, 0, 0), 1)

    def get_frame(self):
        """
        Get the gui frame
        :return: gui frame
        """
        # get the current training image
        x_img = self.get_x_image()
        # get the colorized training target matrix
        y_img = self.get_y_image()
        # apply a filter to see the training image
        # behind the colorized training target matrix
        alpha = 0.3
        y_img = cv2.addWeighted(x_img, alpha, y_img, 1 - alpha, 0)

        params_img = self.get_params_image()

        # get the associated color to the current channel
        color = self.hex2tuple(self.colors[self.channel], normalize=True)

        # draw each reference points
        for pt in self.ref_p:
            cv2.circle(x_img, tuple(pt), 2, color, cv2.FILLED)

        # draw the diameter of the circle if it is the channel mode shape
        # and there is one ref point
        if self.shapes[self.channel] == 2 and len(self.ref_p):
            self.draw_circle_visualization(x_img, y_img, color)

        # draw the brush if the right mouse button is pressed
        if self.brush:
            self.draw_brush(x_img, y_img, color)

        # if the zoom is activated
        if self.zoom_factor > 1:
            self.draw_zoom_window(x_img, y_img)

        # concatenate the training image and the target image horizontally
        concat_xy = np.hstack((x_img, y_img))
        # concatenate vertically with parameter bar image
        gui_img = np.vstack((concat_xy, params_img))
        return gui_img

    def run(self):
        """
        Run the gui until quit it
        """
        while True:
            # Set the window as normal
            cv2.namedWindow("GUI", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("GUI", self.click_event)

            # Set the window in full screen
            cv2.setWindowProperty("GUI", 1, 1)
            # Display the current gui frame
            cv2.imshow("GUI", self.get_frame())

            # Continuously wait for a pressed key
            key = cv2.waitKey(1) & 0xFF

            # activate/deactivate the zoom in the gui and its factor
            if key == keys.zoom:
                # Switch between zoom factors
                self.zoom_factor = 1 + self.zoom_factor % 5
            # if the enter key is pressed for an unlimited contour target
            # draw it from references points
            if key == keys.validate and not self.shapes[self.channel]:
                self.set_poly()
            # remove the last reference point by pressing the return key
            elif key == keys.undo:
                self.ref_p = self.ref_p[:-1] if len(self.ref_p) else []
            # activate/deactivate the brush eraser
            elif key == keys.brush:
                self.brush = not self.brush
            # go to the next image
            elif key == keys.next_image:
                self.update_image(1)
            # go to the previous image
            elif key == keys.previous_image:
                self.update_image(-1)
            # go to the next channel
            elif key == keys.next_channel:
                self.update_channel(1)
            # go to the previous channel
            elif key == keys.previous_channel:
                self.update_channel(-1)
            # delete the current image and its target
            elif key == keys.delete:
                self.delete()
            # exit the gui if the 'q' key is pressed
            elif key == keys.quit:
                # think to save before to quit
                self.save()
                break


if __name__ == "__main__":
    ms = ManualSegmentation(args.x_save_dir, args.y_save_dir, args.config_path)
    ms.run()

