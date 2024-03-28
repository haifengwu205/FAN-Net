import numpy as np
from skimage import io, morphology
from utils import densecrf



def morphology_pre(initial_decisionmap_path, final_decisionamp_path):
    # read image
    image = io.imread(initial_decisionmap_path, as_gray=True)

    # closed operation (expansion first and then corrosion)
    # kernel = morphology.disk(1)
    # image = morphology.closing(image, kernel)
    # image = morphology.dilation(image, kernel)

    # delete small areas
    image_bwareopen = morphology.remove_small_holes(image, area_threshold=100, connectivity=1)
    # inverse
    image_bwareopen = ~image_bwareopen
    # delete small areas
    image_bwareopen = morphology.remove_small_holes(image_bwareopen, area_threshold=80, connectivity=1)
    # inverse
    image_bwareopen = ~image_bwareopen

    # closed operation (expansion first and then corrosion)
    kernel = morphology.disk(7)
    # image_close = morphology.closing(image_bwareopen, kernel)
    image_bwareopen = morphology.dilation(image_bwareopen, kernel)

    # Open operation (corrosion first and then expansion)
    kernel = morphology.disk(5)
    # image_open = morphology.opening(image_close, kernel)
    image_bwareopen = morphology.erosion(image_bwareopen, kernel)

    # delete small areas
    image_bwareopen = morphology.remove_small_holes(image_bwareopen, area_threshold=3600, connectivity=1)
    # inverse
    image_bwareopen = ~image_bwareopen
    # delete small areas
    image_bwareopen = morphology.remove_small_holes(image_bwareopen, area_threshold=3600, connectivity=1)
    # inverse
    image_bwareopen = ~image_bwareopen

    # show
    # print(image_bwareopen[0:10, 0:10] + 0)
    # io.imshow(image_bwareopen)
    # io.show()
    image_bwareopen = image_bwareopen + 0
    image_bwareopen[image_bwareopen < 0.5] = 0
    image_bwareopen[image_bwareopen >= 0.5] = 255

    # save
    io.imsave(final_decisionamp_path, image_bwareopen.astype(np.uint8))


def crf(image1_path, final_decisionmap_save_path, final_crf_path, sdims1, compat1, sdims2, schan, compat2):
    densecrf.crf5(image1_path, final_decisionmap_save_path, final_crf_path,  sdims1, compat1, sdims2, schan, compat2)


def fusion(final_decisionmap_path, image1_path, image2_path, fusion_image_save_path):
    global final_decisionmap
    decisionmap = io.imread(final_decisionmap_path)
    decisionmap = decisionmap / 255.0
    decisionmap[decisionmap < 0.5] = 0
    decisionmap[decisionmap >= 0.5] = 1
    image1 = io.imread(image1_path)
    image2 = io.imread(image2_path)

    # print("image.shape: ", np.array(img1).shape, "decision.shape: ", np.array(initial_decisionmap).shape)
    if len(np.array(image1).shape) == 3:
        final_decisionmap = np.ones([image1.shape[0], image1.shape[1], image1.shape[2]])
        if len(np.array(decisionmap).shape) == 2:
            final_decisionmap[:, :, 0] = decisionmap
            final_decisionmap[:, :, 1] = decisionmap
            final_decisionmap[:, :, 2] = decisionmap
        else:
            final_decisionmap = decisionmap
    else:
        if len(np.array(decisionmap).shape) == 2:
            final_decisionmap = decisionmap
        else:
            final_decisionmap = decisionmap[:, :, 0]

    fusion_image = image1 * final_decisionmap + image2 * (1 - final_decisionmap)
    # save
    io.imsave(fusion_image_save_path, fusion_image.astype(np.uint8))


def fusion_main(initial_decisionmap_path, image1_path, image2_path, final_decisionmap_save_path, final_crf_path,
                fusion_image_save_path, style = 2, sdims1=6, compat1=6, sdims2=50, schan=13, compat2=6):

    if style == 1:
        # morphology
        morphology_pre(initial_decisionmap_path, final_decisionmap_save_path)
        # noly use morphology
        fusion(final_decisionmap_save_path, image1_path, image2_path, fusion_image_save_path)
    elif style == 2:
        # CRF
        crf(image1_path, initial_decisionmap_path, final_crf_path, sdims1, compat1, sdims2, schan, compat2)
        # only use CRF
        fusion(final_crf_path, image1_path, image2_path, fusion_image_save_path)
    elif style == 3:
        # morphology
        morphology_pre(initial_decisionmap_path, final_decisionmap_save_path)
        # CRF
        crf(image1_path, final_decisionmap_save_path, final_crf_path)
        # morphology+CRF
        fusion(final_crf_path, image1_path, image2_path, fusion_image_save_path)


