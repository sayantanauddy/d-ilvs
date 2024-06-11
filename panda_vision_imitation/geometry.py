import numpy as np
from typing import Tuple
import math
import cv2

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, p):
        """
        Adds 2 points
        """
        if isinstance(p, Point):
            x = self.x + p.x
            y = self.y + p.y
            return Point(x,y)
        else:
            raise TypeError('Both operand must be of type Point')

    def __floordiv__(self, num: int):
        """
        Divides a point by an integer
        """
        if isinstance(num, int):
            x = self.x // num
            y = self.y // num
            return Point(x,y)
        else:
            raise TypeError('Second operand must be of type int')

    def __truediv__(self, num):
        """
        Divides a point by a float and rounds the result to int
        """
        if isinstance(num, float) or isinstance(num, int):
            x = round(self.x / float(num))
            y = round(self.y / float(num))
            return Point(x,y)
        else:
            raise TypeError('Second operand must be of type int or float')

    def __repr__(self):
        """
        Prints the Point object
        """
        return f'(x={self.x},y={self.y})'

class Box():
    def __init__(self) -> None:
        """
        Box coordinates starting from top left and 
        going in clockwise direction (0->1->2->3)
        """
        self.p0 = None
        self.p1 = None
        self.p2 = None
        self.p3 = None

        # Centroid
        self.pmid = None

        # Colors for drawing this box
        self.color_p0_p1 = (255,0,0) 
        self.color_p1_p2 = (0,255,0) 
        self.color_p2_p3 = (0,0,255) 
        self.color_p3_p0 = (0,255,255) 
        
    @classmethod
    def init_from_minmax(cls, 
                         pmin: Point, 
                         pmax: Point,
                         ) -> None:
        """
        Init from pmin=(xmin,ymin) and pmax=(xmax,ymax)
        """
        box = cls()
        box.p0 = pmin
        box.p1 = Point(pmin.x, pmax.y)
        box.p2 = pmax
        box.p3 = Point(pmax.x, pmin.y)

        # Set the tracked points
        box.set_tracked_points()

        # Set the centroid
        box.pmid = box.get_mid()

        return box

    def get_mid(self) -> Point:
        """
        Returns centroid of the bbox
        """
        return (self.p0 + self.p2)/2

    def get_x(self) -> Tuple[int,int,int,int]:
        return [self.p0.x, self.p1.x, self.p2.x, self.p3.x]

    def get_y(self) -> Tuple[int,int,int,int]:
        return [self.p0.y, self.p1.y, self.p2.y, self.p3.y]

    def get_all_xy(self):
        """
        Returns tracked point coordinates [p0.x,p0.y,.....,p3.x,p3.y]
        """
        return [self.tracked_p0.x,self.tracked_p0.y,
                self.tracked_p1.x,self.tracked_p1.y,
                self.tracked_p2.x,self.tracked_p2.y,
                self.tracked_p3.x,self.tracked_p3.y]

    def set_tracked_points(self) -> None:
        """
        Sets the tracked points based on the
        angles made with the centroid
        """

        # x coords of all points
        xx = self.get_x()
        # y coords of all points
        yy = self.get_y()

        # Compute the angle w.r.t the center of visual pattern
        pm = self.get_mid()
        xm = pm.x
        ym = pm.y
        xx4angles = [x - xm for x in xx]
        yy4angles = [y - ym for y in yy]
        self.angles = np.arctan2(yy4angles,xx4angles)

        # Sum of angles must be 0 for a rectangle
        assert np.allclose(np.sum(self.angles), 0.0, atol=1e-3), \
               f'degrees={[math.degrees(a) for a in self.angles]}, sum(rads)={np.sum(self.angles)}'
        
        # Sort xx based on angles
        sorted_xx = [xx[i] for i in np.argsort(self.angles)]

        # Sort yy based on angles
        sorted_yy = [yy[i] for i in np.argsort(self.angles)]

        self.tracked_p0 = Point(sorted_xx[0], sorted_yy[0])
        self.tracked_p1 = Point(sorted_xx[1], sorted_yy[1])
        self.tracked_p2 = Point(sorted_xx[2], sorted_yy[2])
        self.tracked_p3 = Point(sorted_xx[3], sorted_yy[3])
        
    def drawbox(self, img: np.array, lw: int):
        """
        Draw this box on a cv2 image
        """
        cv2.line(img, 
                (self.p0.x, self.p0.y), 
                (self.p1.x, self.p1.y), 
                color=self.color_p0_p1, 
                thickness=lw)
        cv2.line(img, 
                (self.p1.x, self.p1.y), 
                (self.p2.x, self.p2.y), 
                color=self.color_p1_p2, 
                thickness=lw)
        cv2.line(img, 
                (self.p2.x, self.p2.y), 
                (self.p3.x, self.p3.y), 
                color=self.color_p2_p3, 
                thickness=lw)
        cv2.line(img, 
                (self.p3.x, self.p3.y), 
                (self.p0.x, self.p0.y), 
                color=self.color_p3_p0, 
                thickness=lw)

    def __repr__(self):
        """
        Prints the Box object
        """
        return f'[tp0={self.p0}, tp1={self.p1}, tp2={self.p2}, tp3={self.p3}]'
