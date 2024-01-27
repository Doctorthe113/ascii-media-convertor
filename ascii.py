import cv2
import click
import time
import os
import numpy as np
from vidgear.gears import WriteGear

import contextlib
with contextlib.redirect_stdout(None):
    import pygame


pygame.init()


DENSITY = np.array([" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"],
                   dtype=str)
SHARPNESS_KERNAL = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")
font = pygame.font.Font('./assets/Inconsolata_Expanded-Black.ttf', 9)


def ascii_ify_img(input_path, output_path):
    image = cv2.imread(input_path)
    height, width = image.shape[:2]
    ASPECT_RATIO = width / height
    RESIZE_IMG_SIZE_H = 108  # 1080 / 10 = 108
    RESIZE_IMG_SIZE_W = np.ceil(RESIZE_IMG_SIZE_H * ASPECT_RATIO).astype(int)

    ASCII_IMG_H = 1080  # all frames will 1080px in height
    ASCII_IMG_SIZE = (np.ceil(ASCII_IMG_H * ASPECT_RATIO).astype(int), ASCII_IMG_H)
    screen = pygame.display.set_mode(ASCII_IMG_SIZE, flags=pygame.HIDDEN)

    img = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    img = cv2.resize(src=img, 
                     dsize=(RESIZE_IMG_SIZE_W, 
                            RESIZE_IMG_SIZE_H))
    img = cv2.filter2D(img, -1, SHARPNESS_KERNAL)

    indices = np.floor(0.0353 * img).astype(int)
    asciiArr = DENSITY[indices.ravel()]
    asciiimgArr = np.reshape(a=asciiArr,
                             newshape=(RESIZE_IMG_SIZE_H, RESIZE_IMG_SIZE_W))
    asciiimgLines = [" ".join(row) for row in asciiimgArr]

    y = 0
    for asciiLine in asciiimgLines:
        ascii_surface = font.render(asciiLine, 
                                    False, 
                                    (255, 255, 255), 
                                    (0, 0, 0))
        ascii_rect = ascii_surface.get_rect()
        ascii_rect.topleft = (0, y)
        screen.blit(ascii_surface, ascii_rect)
        y += font.get_height()
    scrn = pygame.transform.rotate(screen, 90)
    scrn = pygame.transform.flip(scrn, flip_x=False, flip_y=True)
    pygame.display.update()
    asciiRenderArr = pygame.surfarray.array3d(scrn)
    cv2.imwrite(output_path, asciiRenderArr)


@click.command()
@click.option("--input-path", "-i", help="Input video file path")
@click.option("--output-path", "-o", help="Output video file path")
@click.option("--codec", "-c", 
              help="Set the encoder. Run ffmpeg --codec to see the list of available codecs. Default is hevc_amf", 
              default="hevc_amf")
def ascii_ify(input_path, output_path, codec):
    startTime = time.time()
    click.echo("Thanks for using ascii-media-converter. :)")
    click.echo("Starting...")

    if str(input_path).endswith(IMG_EXTENSIONS):
        ascii_ify_img(input_path, output_path)
        endTime = time.time()
        click.echo(f"Total time taken: {endTime - startTime}s")
        return None

    PATH_VARIABLES = os.getenv("Path").split(";")
    for i in PATH_VARIABLES:
        if "ffmpeg" in i:
            FFMPEG_PATH = i


    vid = cv2.VideoCapture(filename=input_path)
    fps = vid.get(propId=cv2.CAP_PROP_FPS)
    frames = vid.get(propId=cv2.CAP_PROP_FRAME_COUNT)
    output_params = {"-vcodec": codec,
                    "-crf": 40,
                    "-preset": "fast",
                    "-input_framerate": fps}
    vidOut = WriteGear(output=output_path,
                    compression_mode=True,
                    logging=False,
                    custom_ffmpeg=FFMPEG_PATH,
                    **output_params)


    _, f = vid.read()
    height, width = f.shape[:2]
    ASPECT_RATIO = width / height
    RESIZE_FRM_SIZE_H = 108  # 1080 / 10 = 108
    RESIZE_FRM_SIZE_W = np.ceil(RESIZE_FRM_SIZE_H * ASPECT_RATIO).astype(int)

    ASCII_FRM_H = 1080  # all frames will 1080px in height
    ASCII_FRM_SIZE = (np.ceil(ASCII_FRM_H * ASPECT_RATIO).astype(int), ASCII_FRM_H)
    screen = pygame.display.set_mode(ASCII_FRM_SIZE, flags=pygame.HIDDEN)
    font = pygame.font.Font('./assets/Inconsolata_Expanded-Black.ttf', size=9)


    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        frm = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
        frm = cv2.resize(src=frm, 
                        dsize=(RESIZE_FRM_SIZE_W, 
                               RESIZE_FRM_SIZE_H))
        frm = cv2.filter2D(src=frm, 
                           ddepth=-1, 
                           kernel=SHARPNESS_KERNAL)

        indices = np.floor(0.0353 * frm).astype(int)
        asciiArr = DENSITY[indices.ravel()]
        asciiFrmArr = np.reshape(a=asciiArr,
                                 newshape=(RESIZE_FRM_SIZE_H, 
                                           RESIZE_FRM_SIZE_W))
        asciiFrmLines = [" ".join(row) for row in asciiFrmArr]

        y = 0
        for asciiLine in asciiFrmLines:
            ascii_surface = font.render(asciiLine, 
                                        False, 
                                        (255, 255, 255), 
                                        (0, 0, 0))
            ascii_rect = ascii_surface.get_rect()
            ascii_rect.topleft = (0, y)
            screen.blit(ascii_surface, ascii_rect)
            y += 10  # font.get_height()
        scrn = pygame.transform.rotate(screen, 90)
        scrn = pygame.transform.flip(scrn, flip_x=False, flip_y=True)
        pygame.display.update()
        asciiRenderArr = pygame.surfarray.array3d(scrn)
        vidOut.write(asciiRenderArr)


    vid.release()
    vidOut.close()
    pygame.quit()

    endTime = time.time()
    click.echo(f"Total time taken: {endTime - startTime}s")
    click.echo(f"FPS: {frames / (endTime - startTime)}")


if __name__ == "__main__":
    ascii_ify()