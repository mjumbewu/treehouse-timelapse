from click import command, argument
from colorutils import color_distance_sq, kmeans, pixel_distance_sq
from functools import partial
from PIL import Image, ImageCms
import sys


@command()
@argument('infilename')
@argument('cluster_count', type=int)
def main(infilename, cluster_count):
    im = Image.open(infilename)
    w, h = im.width, im.height

    # Convert to Lab
    if im.mode != "RGB":
      im = im.convert("RGB")

    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile  = ImageCms.createProfile("LAB", 5500)

    rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
    lab2rgb_transform = ImageCms.buildTransformFromOpenProfiles(lab_profile, srgb_profile, "LAB", "RGB")
    im = ImageCms.applyTransform(im, rgb2lab_transform)

    # Create a data structure from each
    data = tuple(
        (x, y, im.getpixel((x, y)))
        for x in range(w)
        for y in range(h)
    )

    print(str(min(pixel[2] for pixel in data)))
    print(str(max(pixel[2] for pixel in data)))

    # We want it so that the maximum distance along any given axis is about the
    # same. Specifically, we want the color distance to not have a greater effect
    # than the larger of the geometric distances.
    MAX_COLOR_DIST_SQ = color_distance_sq((0,0,0), (255,255,255))
    MAX_SIDE_LENGTH = max(w, h)

    # Scale the colors to have the same distance effect as the geometric
    # dimensions.
    color_dist_sq_scale = MAX_SIDE_LENGTH**2 / MAX_COLOR_DIST_SQ
    color_dist_sq_scale = 1 / 1000000000000000000

    # Get the clusters
    clustered_data, centroids = kmeans(cluster_count, data, color_dist_sq_scale)

    # Save the separate clusters.
    palette = Image.new('LAB', (w, h))
    for cluster_index, centroid in enumerate(centroids):
        im = Image.new('LAB', (w, h))
        for pixel, ci in clustered_data:
            if ci == cluster_index:
                x, y, color = pixel
                im.putpixel((x, y), color)
        with open(f'cluster{cluster_index + 1}of{cluster_count}.png', 'wb') as outfile:
            im = ImageCms.applyTransform(im, lab2rgb_transform)
            im.save(outfile)

    # Save the composite palette image.
    im = Image.new('LAB', (w, h))
    for pixel, ci in clustered_data:
        centroid = centroids[ci]
        x, y, _ = pixel
        _, _, avgcolor = centroid
        im.putpixel((x, y), avgcolor)
    with open(f'{cluster_count}clusterpalette.png', 'wb') as outfile:
        im = ImageCms.applyTransform(im, lab2rgb_transform)
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
