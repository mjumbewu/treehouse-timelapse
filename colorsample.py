from click import command, argument
from colorutils import color_distance_sq, kmeans, pixel_distance_sq
from functools import partial
from PIL import Image
import sys


@command()
@argument('infilename')
@argument('cluster_count', type=int)
def main(infilename, cluster_count):
    im = Image.open(infilename)
    w, h = im.width, im.height

    # Create a data structure from each
    data = tuple(
        (x, y, im.getpixel((x, y)))
        for x in range(w)
        for y in range(h)
    )

    # We want it so that the maximum distance along any given axis is about the
    # same. Specifically, we want the color distance to not have a greater effect
    # than the larger of the geometric distances.
    MAX_COLOR_DIST_SQ = color_distance_sq((0,0,0), (255,255,255))
    MAX_SIDE_LENGTH = max(w, h)

    # Scale the colors to have the same distance effect as the geometric
    # dimensions.
    color_dist_sq_scale = MAX_SIDE_LENGTH**2 / MAX_COLOR_DIST_SQ

    # Get the clusters
    clustered_data, centroids = kmeans(cluster_count, data, color_dist_sq_scale)

    # Save the separate clusters.
    palette = Image.new('RGB', (w, h))
    for cluster_index, centroid in enumerate(centroids):
        im = Image.new('RGB', (w, h))
        for pixel, ci in clustered_data:
            if ci == cluster_index:
                x, y, color = pixel
                im.putpixel((x, y), color)
        with open(f'cluster{cluster_index + 1}of{cluster_count}.png', 'wb') as outfile:
            im.save(outfile)

    # Save the composite palette image.
    im = Image.new('RGB', (w, h))
    for pixel, ci in clustered_data:
        centroid = centroids[ci]
        x, y, _ = pixel
        _, _, avgcolor = centroid
        im.putpixel((x, y), avgcolor)
    with open(f'{cluster_count}clusterpalette.png', 'wb') as outfile:
        im.save(outfile)

    # Calculate the sum of the square errors
    sq_error = 0
    for pixel, ci in clustered_data:
        centroid = centroids[ci]
        sq_error += pixel_distance_sq(pixel, centroid, color_dist_sq_scale)

    print(f'Number of clusters: {cluster_count}')
    print(f'Sum of squared error: {sq_error}')


if __name__ == '__main__':
    main()
