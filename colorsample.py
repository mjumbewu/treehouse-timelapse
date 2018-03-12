from functools import partial
from PIL import Image
from random import sample


MAX_COLOR = 255**2


def average(nums):
    nums = tuple(nums)
    return sum(nums) / len(nums)


def pixel_distance_sq(p1, p2):
    x1, y1, c1 = p1
    x2, y2, c2 = p2

    dsq = (
        (x2 - x1)**2 +
        (y2 - y1)**2 +
        color_distance_sq(c1, c2)
    )
    return dsq


def pixel_average(pixels):
    pixels = tuple(pixels)
    avgx = average(p[0] for p in pixels)
    avgy = average(p[1] for p in pixels)
    avgc = color_average(p[2] for p in pixels)

    return (avgx, avgy, avgc)


def color_distance_sq(c1, c2):
    r1, g1, b1 = (x**2 for x in c1)
    r2, g2, b2 = (x**2 for x in c2)

    dsq = (r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2
    return dsq


def color_average(colors):
    colors = tuple(colors)
    avgr = average(c[0]**2 for c in colors)
    avgg = average(c[1]**2 for c in colors)
    avgb = average(c[2]**2 for c in colors)

    return (int(avgr**0.5), int(avgg**0.5), int(avgb**0.5))


def kmeans(k, data):
    """
    Step 1 - Pick K random points as cluster centers called centroids.
    Step 2 - Assign each xi to nearest cluster by calculating its distance to each centroid.
    Step 3 - Find new cluster center by taking the average of the assigned points.
    Step 4 - Repeat Step 2 and 3 until none of the cluster assignments change.
    """
    if len(data) < k:
        raise ValueError(f'Not enough data for {k} clusters')

    indexed_centroids = tuple(enumerate(sample(data, k)))
    clustered_data = tuple((pixel, None) for pixel in data)

    def centroid_dist_sq(pixel, indexed_centroid):
        index, centroid = indexed_centroid
        return pixel_distance_sq(pixel, centroid)

    itercount = 0
    while True:

        itercount += 1
        print(f'Iteration #{itercount}; centroids:')
        for index, centroid in indexed_centroids:
            print(f'\t{centroid}')

        new_clustered_data = []
        for pixel in data:
            index, centroid = min(indexed_centroids, key=partial(centroid_dist_sq, pixel))
            new_clustered_data.append((pixel, index))

        if clustered_data == new_clustered_data:
            break

        clustered_data = new_clustered_data
        indexed_centroids = []
        for index in range(k):
            cluster = tuple(pixel for (pixel, i) in clustered_data if i == index)
            print(f'Cluster for centroid {index} has {len(cluster)} pixels')
            centroid = pixel_average(cluster)
            indexed_centroids.append((index, centroid))

    return clustered_data, indexed_centroids


def main():
    # Open the image
    from PIL import Image
    # im = Image.open("/home/mjumbewu/Pictures/LaConcha.jpg")
    im = Image.open("clustertest.bmp")
    w, h = im.width, im.height

    # Create a data structure from each
    data = tuple(
        (x, y, im.getpixel((x, y)))
        for x in range(w)
        for y in range(h)
    )

    # Get the clusters
    count = 4
    clustered_data, indexed_centroids = kmeans(count, data)

    # Save the separate clusters.
    palette = Image.new('RGB', (w, h))
    for c in range(count):
        index, (cx, cy, avgcolor) = indexed_centroids[c]
        im = Image.new('RGB', (w, h))
        for pixel, i in clustered_data:
            if i == c:
                x, y, color = pixel
                im.putpixel((x, y), color)
                palette.putpixel((x, y), tuple(int(n) for n in avgcolor))

        with open(f'cluster{c}.png', 'wb') as outfile:
            im.save(outfile)

    with open('clusterpalette.png', 'wb') as outfile:
        palette.save(outfile)


if __name__ == '__main__':
    main()
