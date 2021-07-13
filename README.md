# Beer Challenge
Image Processing Beer Challenge proposed by Visio

More details from the Beer Challenge can be found [here](https://www.notion.so/Proposta-de-Projeto-336e8afb603447109116a61d147c0e09)

## Main objective
This project consists in the creation of a software for automatic counting of yeast cells in microscope images of fermentation samples.

## Input images
The sequence of input images corresponds to the central region of the Neubauer chamber, divided into NxM images, where N is the number of lines and M the number of columns. This NxM sequence must be reconstructed to form a single image.

### Neubauer chamber reference grid:
![Neubauer chamber](./images/neubauer_grid.png)

### Example input images:
All input images were gathered from the [Beer Challenge page](https://www.notion.so/Proposta-de-Projeto-336e8afb603447109116a61d147c0e09)

![Input image 1](./dataset/1Quad.jpg)

![Input image 2](./dataset/2Quad.jpg)

![Input image 3](./dataset/7Quad2.jpg)

![Input image 4](./dataset/8Quad2.jpg)

## Steps

### Grid reconstruction

Each image may have small rotations or areas that overlap with other input images that need to be considered when reconstructing the grid.
To reconstruct the grid, I try and get the biggest contour (grid pattern) on a processed image. Then I resize the contour and rotate the grid to get the output image. Finally, I (WIP) put them together. Only the central part of the Neubauer chamber will be considered.

Here is an example of a reconstructed grid (using only 1 dataset image - 1Quad.jpg)

![Reconstructed grid image](./reconstructed.jpg)

### Cell identification

Identify cells in image. For this step I will probably use image enhancement, segmentation and edge detection.
This [geeks for geeks article](https://www.geeksforgeeks.org/python-blood-cell-identification-using-image-processing/) looks promising to help me.

### Differentiation of living and dead cells

In the images offered, the main visual difference is the coloration: The living cells are translucent/white, while the dead ones are bluish.
For this step I will mainly use the color channels to differentiate cells. I can split channels, adjust brightness/contrast, add a mean filter, unsharp mask, then threshold.
The contents of this [publication](https://www.researchgate.net/publication/51021408_Digital_Image_Processing_of_LiveDead_Staining) might be used as basis.

### Cell count

Since cells are vibibly different from each other as well as the background, I should be able to use color thresholding. 
>Converting the image to HSV format then use a lower/upper color threshold to create a binary mask. From here we perform morphological operations to smooth the image and remove small bits of noise.

[Stackoverflow question](https://stackoverflow.com/questions/58751101/count-number-of-cells-in-the-image) about same/similar subject.

## Expected output

Output contains four measurements and has the following format:

`TotalNumberOfCells NumberOfLivingCells NumberOfDeadCells %OfLivingCells`

Additionaly, there should be a way to visualize and differentiate living from dead cells.

