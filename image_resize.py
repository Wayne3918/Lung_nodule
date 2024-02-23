from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

img256 = cv2.imread('C:\\Users\\cheny\\Desktop\\FYP\\Final report images\\project\\sofa_256.png',0)
img128 = cv2.imread('C:\\Users\\cheny\\Desktop\\FYP\\Final report images\\project\\sofa_128.png',0)
img64 = cv2.imread('C:\\Users\\cheny\\Desktop\\FYP\\Final report images\\project\\sofa_64.png',0)
img32 = cv2.imread('C:\\Users\\cheny\\Desktop\\FYP\\Final report images\\project\\sofa_32.png',0)
#img1 = cv2.imread('left.png',1)
#img1 = cv2.resize(img1, (797,1212))
#dst1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#cv2.imwrite('r.png',dst1)
cv2.waitKey()

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MAE: %.2f, SSIM: %.2f" % (m, s))
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
	# show the images
	plt.show()

fig = plt.figure("Images")
images = ("Sofa64", img64), ("Sofa32", img32)
# loop over the images
for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(1, 2, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")
# show the figure
plt.show()
# compare the images
compare_images(img64, img64, "64vs64MAE")
compare_images(img64, img32, "64vs32MAE")