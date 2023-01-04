#!/usr/bin/env python

__author__ = "Christopher Hahne"
__email__ = "info@christopherhahne.de"
__license__ = """
    Copyright (c) 2017 Christopher Hahne <info@christopherhahne.de>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

# external libs
import numpy as np


class NonMaxSuppression(object):

    def __init__(self, img):

        # input variables
        self._img = img

        # internal variable
        self._map = np.zeros(self._img.shape, dtype=self._img.dtype)

    @property
    def map(self):
        return self._map

    @property
    def idx(self):
        return np.array(self._map.nonzero())

    def main(self):

        # find local maxima
        self._non_max_suppression()

        # suppress negative local maxima
        self._map[self._map < 0] = 0

    def _non_max_suppression(self):
        """ adaption of non-maximum suppression by Tuan Q. Pham """

        h, w = self._img.shape
        skip = np.zeros([h, 2], dtype=self._img.dtype)
        cur = 0
        next = 1 # scanline masks
        for c in range(1, w-1):
            r = 1
            while r < h-1:
                # skip current pixel
                if skip[r, cur]:
                    r += 1
                    continue
                # compare to pixel on the left
                if self._img[r, c] <= self._img[r+1, c]:
                    r += 1
                    # rising
                    while r < h-1 and self._img[r, c] <= self._img[r+1, c]:
                        r += 1
                    if r == h-1:
                        # reach scanline's local maximum
                        break
                else:
                    # compare to pixel on the right
                    if self._img[r, c] <= self._img[r-1, c]:
                        r += 1
                        continue
                skip[r+1, cur] = 1

                # compare to 3 future then 3 past neighbors
                if self._img[r, c] <= self._img[r-1, c+1]:
                    r += 1
                    continue
                # skip future neighbors only
                skip[r-1, next] = 1
                if self._img[r, c] <= self._img[r, c+1]:
                    r += 1
                    continue
                skip[r, next] = 1
                if self._img[r, c] <= self._img[r+1, c+1]:
                    r += 1
                    continue
                skip[r+1, next] = 1
                if self._img[r, c] <= self._img[r-1, c-1]:
                    r += 1
                    continue
                if self._img[r, c] <= self._img[r, c-1]:
                    r += 1
                    continue
                if self._img[r, c] <= self._img[r+1, c-1]:
                    r += 1
                    continue

                self._map[r, c] = self._img[r, c]
                # a new local maximum is found
                r += 1

            tmp = cur
            cur = next
            # swap mask indices
            next = tmp
            # reset next scanline mask
            skip[:, next] = 0

        return True
