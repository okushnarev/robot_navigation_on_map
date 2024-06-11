import cv2 as cv
import numpy as np
from pathlib import Path



if __name__ == '__main__':
    import_dir = Path('./images')

    export_dir = Path('./maps')
    export_dir.mkdir(exist_ok=True, parents=True)

    for img_path in import_dir.glob('*.jpg'):
        img_rgb = cv.imread(img_path)
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)


        img_gray = cv.GaussianBlur(img_gray, (7, 7), 0)
        thresh_adaptive = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        kernel = np.ones((9, 9), np.uint8)
        thresh_adaptive = cv.morphologyEx(thresh_adaptive, cv.MORPH_OPEN, kernel)
        _, thresh_otsu = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        _, thresh_global = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)

        contours, _ = cv.findContours(thresh_otsu, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours):
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.drawContours(img_rgb, contours, -1, (0, 0, 255), 2)

        img_name = img_path.name[:-4]
        cv.imwrite(export_dir / f'{img_name}.png', thresh_otsu)
