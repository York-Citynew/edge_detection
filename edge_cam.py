import numpy as np
import cv2


def std_filter(img, ksize): return np.sqrt(
    cv2.boxFilter(img**2, -1, (ksize, ksize)) -
    cv2.boxFilter(img, -1, (ksize, ksize))**2
)


def zero_crossing(img, thrsh):
    img_shrx = img.copy()
    img_shrx[:, 1:] = img_shrx[:, :-1]
    img_shdy = img.copy()
    img_shdy[1:, :] = img_shdy[:-1, :]
    res = (img == 0) | (img * img_shrx < 0) | (img * img_shdy < 0)
    std_image = std_filter(img, 3) / img.max()
    res = res & (std_image > thrsh)
    return np.uint8(res)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    mode = 'o'
    sigma = 3
    sobel_th = 100
    canny_l, canny_h = 50, 150

    while True:
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgb = cv2.GaussianBlur(img, (sigma, sigma), 0)
        if mode == 'o':
            # res = the original image
            res = img
        elif mode == 'x':
            # res = Sobel gradient in x direction
            res = np.abs(cv2.Sobel(imgb, cv2.CV_64F, 1, 0, ksize=sigma))
        elif mode == 'y':
            res = np.abs(cv2.Sobel(imgb, cv2.CV_64F, 0, 1, ksize=sigma))
            # res = Sobel gradient in y direction
        elif mode == 'm':
            res = np.sqrt(cv2.Sobel(imgb, cv2.CV_64F, 0, 1, ksize=sigma) **
                          2 + cv2.Sobel(imgb, cv2.CV_64F, 1, 0, ksize=sigma)**2)
            # res = magnitude of Sobel gradient
        elif mode == 's':
            res = np.where(
                np.sqrt(
                    cv2.Sobel(imgb, cv2.CV_64F, 0, 1)**2 +
                    cv2.Sobel(imgb, cv2.CV_64F, 0, 1)**2
                ) > sobel_th, 1, 0)
            # res = Sobel + thresholding edge detection
        elif mode == 'l':
            res = zero_crossing(cv2.Laplacian(
                imgb, cv2.CV_64F, ksize=sigma), .1)
            # res = Laplacian edges
        elif mode == 'c':
            res = cv2.Canny(imgb, canny_l, canny_h)
        res = res.astype(np.float64) / res.max()
        res = cv2.putText(res, mode, (0, 50), 0, 1, (255, 255, 255), 2)
        cv2.imshow("my stream", res)

        key = chr(cv2.waitKey(1) & 0xFF)

        if key in ['o', 'x', 'y', 'm', 's', 'c', 'l']:
            mode = key
        if key == '-' and sigma > 1:
            sigma -= 2
            print(f"sigma = {sigma}")
        elif key in ['+', '=']:
            sigma += 2
            print(f"sigma = {sigma}")
        elif key == 'u':
            sobel_th += 1
            print(f"sobel_th = {sobel_th}")
        elif key == 'd':
            if (new_sobel_th := sobel_th - 1) > 0:
                sobel_th = new_sobel_th
            print(f"sobel_th = {sobel_th}")
        elif key == '.':
            canny_h += 1
            print(f"canny_h = {canny_h}")
        elif key == ',':
            if (new_canny_h := canny_h - 1) > 0:
                canny_h = new_canny_h
            print(f"canny_h = {canny_h}")
        elif key == '6':
            canny_l += 1
            print(f"canny_l = {canny_l}")
        elif key == '4':
            if (new_canny_l := canny_l - 1) > 0:
                canny_l = new_canny_l
            print(f"canny_l = {canny_l}")
        elif key == 'q':
            break

    cap.release()
    cv2.destroyAllWindows()
