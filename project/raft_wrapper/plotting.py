import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


def plot_torch_images(images, **imshow_kwargs):
    plt.rcParams["savefig.bbox"] = "tight"
    if not isinstance(images[0], list):
        # Make a 2d grid even if there's just 1 row
        images = [images]

    num_rows = len(images)
    num_cols = len(images[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(images):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()

    # get image and convert to numpy
    fig = plt.gcf()
    fig.canvas.draw()
    np_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    np_img = np_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return np_img
