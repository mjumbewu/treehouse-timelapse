from math import ceil, floor
from multiprocessing import Pool, Queue
from os import listdir
from os.path import join as pathjoin
from PIL import Image
import re


SNAP_FOLDER = '/home/mjumbewu/Pictures/malcolmxcam'
THUMB_WIDTH = 75
THUMB_HEIGHT = None
THUMB_MODE = 1


def resize_snap(image_metadata):
    """
    Resize a snapshot to a thumbnail.

    image_metadata is a tuple of:
    - filename
    - (iso date, hour, sequence num within the hour)
    - n-th day since the earliest in the set
    """
    f, (day, hour, seq), daynum = image_metadata

    # Open the snapshot and resize it down
    snap = Image.open(pathjoin(SNAP_FOLDER, f))

    if THUMB_MODE == 1:
    # OPTION 1: Resize the snapshot and just returned the resized image directly
    #           as the thumbnail.

        thumb = snap.resize((THUMB_WIDTH, THUMB_HEIGHT))

    # ============
    # END OPTION 1

    elif THUMB_MODE == 2:
    # OPTION 2: Resize the snapshot to half the dimensions of the thumbnail, and
    #           flip the image vertically and horizontally to create smoother
    #           transitions between the edges of the thumbnails.

        thumb_quarter = snap.resize((ceil(THUMB_WIDTH / 2), ceil(THUMB_HEIGHT / 2)))
        thumb = Image.new('RGB', (THUMB_WIDTH, THUMB_HEIGHT))

        thumb.paste(thumb_quarter)
        thumb.paste(thumb_quarter.transpose(Image.FLIP_LEFT_RIGHT), (floor(THUMB_WIDTH / 2), 0))
        thumb.paste(thumb_quarter.transpose(Image.FLIP_TOP_BOTTOM), (0, floor(THUMB_HEIGHT / 2)))
        thumb.paste(thumb_quarter.transpose(Image.ROTATE_180), (floor(THUMB_WIDTH / 2), floor(THUMB_HEIGHT / 2)))

    # ============
    # END OPTION 2

    else:
        raise ValueError(f'Invalid THUMB_MODE: {THUMB_MODE}')

    # Determine the coordinates at which to paste in the thumbnail
    col = hour * 6 + seq
    row = daynum
    coords = (col * THUMB_WIDTH, row * THUMB_HEIGHT)

    return (thumb, coords)


def iter_image_metadata(days, indexed_filenames):
    """
    Generator for image metadata. Yields a tuple of:
    - filename
    - image key
    - day index

    In other words, the same parameters as are consumed by `resize_snap`
    """
    for daynum, day in enumerate(days):
        for hour in range(24):
            for seq in range(6):
                key = (day, hour, seq)
                try:
                    f = indexed_filenames[key]
                except KeyError:
                    print(f'Couldn\'t find file for {key}')
                    continue
                yield f, key, daynum


def main():
    global THUMB_WIDTH
    global THUMB_HEIGHT

    if not any([THUMB_WIDTH, THUMB_HEIGHT]):
        raise ValueError('One of width or height must be set.')

    # Aspect ratio is 4x3
    if not THUMB_WIDTH:
        THUMB_WIDTH = int(THUMB_HEIGHT * 4 / 3)
    if not THUMB_HEIGHT:
        THUMB_HEIGHT = int(THUMB_WIDTH * 3 / 4)


    # Read and index images by timestamp
    pattern = re.compile(r'2018-\d\d-\d\dT\d\d:\d\d:\d\d\+\d\d:\d\d\.jpg')
    image_filenames = [f for f in listdir(SNAP_FOLDER) if pattern.match(f)]

    days = list(sorted(set(f[:10] for f in image_filenames)))
    print(f'Found data for {len(days)} days.')

    filekey = lambda f: (f[:10], int(f[11:13]), int(f[14:16])/10)
    indexed_filenames = {
        filekey(f): f
        for f in image_filenames
    }

    # Create a pool of workers to resize images
    pool = Pool(7)

    # Farm out the image resizing to the worker pool
    images_metadata = iter_image_metadata(days, indexed_filenames)
    thumb_iter = pool.imap(resize_snap, images_metadata)

    # Calculate the full size of the grid canvas. There are 24 hours, 6 frames per
    # hour for 144 frames per day.
    frames_per_day = 24 * 6
    num_days = len(days)

    canvas_width = THUMB_WIDTH * frames_per_day
    canvas_height = THUMB_HEIGHT * num_days

    # Create the canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height))

    # Paste in all the thumb images
    for thumb, coords in thumb_iter:
        canvas.paste(thumb, coords)

    # Save the canvas
    with open('outfile.jpg', 'wb') as outfile:
        canvas.save(outfile)


if __name__ == '__main__':
    main()
