""" Common utilities for MNIST image classification task.

"""
import numpy as np
import matplotlib.pyplot as plt

def show_mnist_img(img_data, img_data_meta, img_height=28, img_width=28):
    assert len(img_data) == img_height * img_width
    t_img = process_mnist_img(img_data)
    plt.imshow(t_img, cmap='gray')
    plt.title('y: {0}, y_hat: {1}, conf for y: {2}'.format(img_data_meta['y'], 
        img_data_meta['y_hat'], img_data_meta['conf']))

def plot_gallery(img_data, img_data_pred, n_row=3, n_col=4):
    """Plot a gallery of n_rows * n_cols of mnist images.
    
    Parameters
    ----------
    img_data : List of arrays
        List of mnist image data arrays.
    img_data_pred : List of dicts
        List of dicts which contains the predictions and actual y.
    n_row : int (optional)
        number of rows needed in gallery. Default : 3
    n_col : int (optional)
        number of columns needed in gallery. Default : 4

    """
    assert len(img_data) == n_row * n_col 
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.80)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        t_img = process_mnist_img(img_data[i])
        plt.imshow(t_img, cmap='gray')
        plt.title('y: {0}, y_hat: {1}\ny_hat conf : {2:.3f}'.format(img_data_pred[i]['y'], 
            img_data_pred[i]['y_hat'], img_data_pred[i]['conf'][0]))
        plt.xticks(())
        plt.yticks(())

def process_mnist_img(img_data, img_height=28, img_width=28):
    """Process the image data so that it can be displayed.
    
    Parameters
    ----------
    img_data: List
        image data
    img_height: int (optional)
        height of the image. Default: 28
    img_width: int 
        width of thee image. Default: 28

    Returns
    -------
    trg_img: List

    """
    assert len(img_data) == img_height * img_width
    trg_img = (np.array(img_data, dtype='float')).reshape(img_height, img_width)
    trg_img = np.uint8(trg_img * 255)
    return trg_img


