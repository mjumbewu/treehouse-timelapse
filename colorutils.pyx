#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=False

from functools import partial
from random import sample

from cpython cimport array
from array import array

def average(nums):
    nums = tuple(nums)
    return sum(nums) / len(nums)


cdef double _pixel_distance_sq(double x1, double y1, long r1, long g1, long b1,
                               double x2, double y2, long r2, long g2, long b2,
                               double color_dist_sq_scale):
    dsq = (
        (x2 - x1)**2 +
        (y2 - y1)**2 +
        _color_distance_sq(r1, g1, b1, r2, g2, b2) / color_dist_sq_scale
    )
    return dsq

def pixel_distance_sq(p1, p2, color_dist_sq_scale=1):
    x1, y1, (r1, g1, b1) = p1
    x2, y2, (r2, g2, b2) = p2
    return _pixel_distance_sq(x1, y1, r1, g1, b1,
                              x2, y2, r2, g2, b2,
                              color_dist_sq_scale)


def pixel_average(pixels):
    pixels = tuple(pixels)
    avgx = average(p[0] for p in pixels)
    avgy = average(p[1] for p in pixels)
    avgc = color_average(p[2] for p in pixels)

    return (avgx, avgy, avgc)


cdef long _color_distance_sq(long r1, long g1, long b1,
                             long r2, long g2, long b2):
    # r1 = r1**2
    # g1 = g1**2
    # b1 = b1**2
    # r2 = r2**2
    # g2 = g2**2
    # b2 = b2**2

    cdef long dsq = (r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2
    return dsq

def color_distance_sq(c1, c2):
    return _color_distance_sq(c1[0], c1[1], c1[2], c2[0], c2[1], c2[2])


def color_average(colors):
    colors = tuple(colors)
    avgr = average(c[0] for c in colors)
    avgg = average(c[1] for c in colors)
    avgb = average(c[2] for c in colors)

    return (int(avgr), int(avgg), int(avgb))


def kmeans(int k, data, double color_dist_sq_scale=1.0):
    """
    Step 1 - Pick K random points as cluster centers called centroids.
    Step 2 - Assign each xi to nearest cluster by calculating its distance to each centroid.
    Step 3 - Find new cluster center by taking the average of the assigned points.
    Step 4 - Repeat Step 2 and 3 until none of the cluster assignments change.
    """
    if len(data) < k:
        raise ValueError(f'Not enough data for {k} clusters')

    cdef int num_pixels = len(data)

    # Populate C arrays with pixel data.
    # -- Coordinates for pixel data
    cdef double [:] xs = array('d', [pixel[0] for pixel in data])
    cdef double [:] ys = array('d', [pixel[1] for pixel in data])
    # -- Colors for pixel data
    cdef long [:] rs = array('l', [pixel[2][0] for pixel in data])
    cdef long [:] gs = array('l', [pixel[2][1] for pixel in data])
    cdef long [:] bs = array('l', [pixel[2][2] for pixel in data])

    # Find initial centroids. Stick them in C arrays too.
    centroids = tuple(sample(data, k))
    # -- Coordinates for cluster centroids
    cdef double [:] centroid_xs = array('d', [centroid[0] for centroid in centroids])
    cdef double [:] centroid_ys = array('d', [centroid[1] for centroid in centroids])
    # -- Colors for cluster centroids
    cdef long [:] centroid_rs = array('l', [centroid[2][0] for centroid in centroids])
    cdef long [:] centroid_gs = array('l', [centroid[2][1] for centroid in centroids])
    cdef long [:] centroid_bs = array('l', [centroid[2][2] for centroid in centroids])

    # Create index counters for clusters and pixels
    cdef int ci, pi

    # Create variables for the distance from pixels to their cluster's centroid
    cdef int min_dist_ci
    cdef double min_dist

    # Initialize the cluster assignments for each pixel
    cdef int [:] cis = array('i', [-1] * num_pixels)
    cdef int [:] new_cis = array('i', [-1] * num_pixels)

    itercount = 0
    while True:

        itercount += 1
        print(f'Iteration #{itercount}; centroids:')
        for centroid in centroids:
            print(f'\t{centroid}')

        # Populate arrays with the centroid data
        for ci, centroid in enumerate(centroids):
            centroid_xs[ci] = centroid[0]
            centroid_ys[ci] = centroid[1]
            centroid_rs[ci] = centroid[2][0]
            centroid_gs[ci] = centroid[2][1]
            centroid_bs[ci] = centroid[2][2]

        # Find the min distanced cluster for each pixel
        for pi in range(num_pixels):
            min_dist_ci = min_dist = -1
            for ci in range(k):
                dist = _pixel_distance_sq(
                    xs[pi], ys[pi], rs[pi], gs[pi], bs[pi],
                    centroid_xs[ci], centroid_ys[ci], centroid_rs[ci], centroid_gs[ci], centroid_bs[ci],
                    color_dist_sq_scale)

                if min_dist_ci == -1 or dist < min_dist:
                    min_dist = dist
                    min_dist_ci = ci

            new_cis[pi] = min_dist_ci

        for pi in range(num_pixels):
            if cis[pi] != new_cis[pi]:
                break # out of this for loop, continuing with the rest of the iteration.
        else:
            break # out of the while loop.

        # Copy the new cluster indexes into the old one.
        for pi in range(num_pixels):
            cis[pi] = new_cis[pi]

        centroids = []
        for ci in range(k):
            cluster = tuple(
                (xs[pi], ys[pi], (rs[pi], gs[pi], bs[pi]))
                for pi in range(num_pixels) if cis[pi] == ci
            )
            print(f'Cluster for centroid {ci} has {len(cluster)} pixels')
            centroid = pixel_average(cluster)
            centroids.append(centroid)

    clustered_data = tuple((data[pi], new_cis[pi]) for pi in range(num_pixels))
    return clustered_data, centroids
