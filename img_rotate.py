import numpy as np
import cv2
from itertools import product
import argparse


def options():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i','--image', default=0, help='Enter the image directory or enter 0 to open the camera')
    arg.add_argument('-s','--speed', type=int, default=2, help='Sets speed')
    arg.add_argument('--effect', type=int, default=1, help='1: On, 0: Off, For real shuffling effect')
    arg.add_argument('--size', type=int, default=None, help='Size of the image, Must be power of 2')

    return arg.parse_args()


class Image:
    def __init__(self, img, l):
        h, w, _ = img.shape
        self.step = int(np.log2(h))

        l = l - 1

        setattr(self, f"level_{l + 1}", [])

        for i, j in product(range(2 ** l), range(2 ** l)):
            block = h // (2 ** (l))
            half = block // 2

            getattr(self, f"level_{l + 1}").append(
                img[i * block + half: (i + 1) * block, j * block:(j + 1) * block - half, :])  # (2, 2)

            getattr(self, f"level_{l + 1}").append(
                img[i * block: (i + 1) * block - half, j * block:(j + 1) * block - half, :])  # (1,1)

            getattr(self, f"level_{l + 1}").append(
                img[i * block: (i + 1) * block - half, j * block + half:(j + 1) * block, :])  # (1, 2)

            getattr(self, f"level_{l + 1}").append(
                img[i * block + half: (i + 1) * block, j * block + half:(j + 1) * block, :])  # (2, 1)


def move(img, patch, index, iter, level, effect=False):
    img_copy = img

    direction = direction_calc(index)

    patch_size = int(img_copy.shape[0] / (2 ** level))

    loch, locw = index_to_loc(index, level)

    dh, dw = direction * iter

    dw = int(dw)
    dh = int(dh)

    if index % 4 == 3 and effect == 1:
        patch_alpha = np.ones_like(patch)

        patch_alpha[0:patch_size - iter, 0:iter] = 0

        ones = np.ones_like(patch)

        first = np.multiply(img_copy[loch * patch_size + dh:(loch + 1) * patch_size + dh,
                            locw * patch_size + dw:(locw + 1) * patch_size + dw], ones - patch_alpha)
        second = np.multiply(patch, patch_alpha)

        img_copy[loch * patch_size + dh:(loch + 1) * patch_size + dh,
        locw * patch_size + dw:(locw + 1) * patch_size + dw] = first + second

    else:
        img_copy[loch * patch_size + dh:(loch + 1) * patch_size + dh,
        locw * patch_size + dw:(locw + 1) * patch_size + dw] = patch
    return img_copy


def direction_calc(index):
    if index % 4 == 0:
        direction = [-1, 0]  # up
    elif index % 4 == 1:
        direction = [0, 1]  # right
    elif index % 4 == 2:
        direction = [1, 0]  # down
    elif index % 4 == 3:
        direction = [0, -1]  # left
    else:
        raise ValueError

    return np.array(direction)


def index_to_loc(index, level):
    _index = index // 4

    _locw = int(_index % (2 ** (level - 1)))
    _loch = int(_index // (2 ** (level - 1)))

    if index % 4 == 0:
        loch = 2 * _loch + 1
        locw = 2 * _locw

    elif index % 4 == 1:
        loch = 2 * _loch
        locw = 2 * _locw

    elif index % 4 == 2:
        loch = 2 * _loch
        locw = 2 * _locw + 1

    elif index % 4 == 3:
        loch = 2 * _loch + 1
        locw = 2 * _locw + 1

    else:
        raise ValueError

    return loch, locw


def camera():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        key = cv2.waitKey(25)

        cv2.imshow("Press s to capture", img)

        if key == ord('s'):
            cap.release()
            cv2.destroyAllWindows()
            return img

    cv2.destroyAllWindows()
    cap.release()
    print("Couldn't open the camera")


def imread(dir, size,camera=False):
    if camera:
        original = dir
    else:
        original = cv2.imread(dir)
        if not isinstance(original, np.ndarray):
            raise FileNotFoundError

    h, w, c = original.shape

    if h != w:
        print("Making it square...")

        if w > h:
            p = (w - h) // 2
            original = original[:, p:p + h]
        elif h > w:
            p = (h - w) // 2
            original = original[p:p + w, :]

    if size is None:
        new_h = int(2 ** np.ceil(np.log2(h))) if int(2 ** np.ceil(np.log2(h))) < 513 else 512
    else:
        new_h=size
    resized = cv2.resize(original, (new_h, new_h), interpolation=cv2.INTER_CUBIC)

    print(f"Resized to {new_h}x{new_h}")

    return resized, new_h


if __name__ == '__main__':

    opt = options()
    speed = opt.speed
    if opt.image == '1':
        print("Camera is opening...")
        img = camera()
        isCamera = True

    else:
        isCamera = False
        img = opt.image
        print("Image is loading...")

    resized, h = imread(img, opt.size, isCamera)

    updated = resized.copy()
    step = int(np.log2(h))

    while True:
        speed = opt.speed
        for l in range(1, step):
            layer = Image(updated, l)
            pathes = getattr(layer, f"level_{l}")
            if l == 4:
                speed = 1

            for iteration in range(0, int(h / (2 ** l)) + 1, speed):
                updated = np.zeros_like(resized)

                for index, patch in enumerate(pathes):
                    updated = move(updated, patch, index, iteration, l, opt.effect)

                cv2.imshow("You spin me right 'round, baby right round like a record, baby", updated)

                key = cv2.waitKey(5)
                if key == ord('q'):
                    exit()
