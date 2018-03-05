from multiprocessing import Pool, Queue
from os import listdir
from os.path import join as pathjoin
from PIL import Image
import re


SNAP_FOLDER = '/home/mjumbewu/Pictures/malcolmxcam'
THUMB_WIDTH = 10
THUMB_HEIGHT = None


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
    thumb = snap.resize((THUMB_WIDTH, THUMB_HEIGHT))

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
                try: f = indexed_filenames[key]
                except KeyError: continue
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

    # Calculate the full size of the grid canvas. There are 24 hours, 6 frames per
    # hour for 144 frames per day.
    frames_per_day = 24 * 6
    num_days = 14

    canvas_width = THUMB_WIDTH * frames_per_day
    canvas_height = THUMB_HEIGHT * num_days

    # Create the canvas
    pattern = re.compile(r'2018-\d\d-\d\dT\d\d:\d\d:\d\d\+\d\d:\d\d\.jpg')
    canvas = Image.new('RGB', (canvas_width, canvas_height))

    # Read and index images by timestamp
    image_filenames = [f for f in listdir(SNAP_FOLDER) if pattern.match(f)]

    days = list(sorted(set(f[:10] for f in image_filenames)))
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

    # Paste in all the thumb images
    for thumb, coords in thumb_iter:
        canvas.paste(thumb, coords)

    # Save the canvas
    with open('outfile.jpg', 'wb') as outfile:
        canvas.save(outfile)


if __name__ == '__main__':
    main()
