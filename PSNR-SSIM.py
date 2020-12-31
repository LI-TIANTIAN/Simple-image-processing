import tensorflow as tf
from SSIM_PIL import compare_ssim
from PIL import Image
import numpy as np


def read_img(path):
	return tf.image.decode_image(tf.read_file(path))

def psnr(tf_img1, tf_img2):
	return tf.image.psnr(tf_img1, tf_img2, max_val=255)

def psnrnew(image_a, image_b):
    image_a_data = np.asarray(image_a).astype('float32')
    image_b_data = np.asarray(image_b).astype('float32')
    diff = image_a_data - image_b_data
    diff = diff.flatten('C')
    rmse = np.math.sqrt(np.mean(diff ** 2.))
    return 20 * np.math.log10(255.0 / rmse)

def _main():
	# t1 = read_img('C:/Users/KEVIN/Desktop/flowers.jpg')
	# t2 = read_img('C:/Users/KEVIN/Desktop/flowers_sr.jpg')
	# t2 = t2.resize(t1.size)
	# with tf.Session() as sess:
	# 	sess.run(tf.global_variables_initializer())
	# 	y = sess.run(psnr(t1, t2))
	# 	print('PSNR:'+str(y))
	image1 = Image.open("C:/Users/Yanfen Li/Desktop/result/ESPCN/1/original.png")
	image2 = Image.open("C:/Users/Yanfen Li/Desktop/result/ESPCN/1/result.png")
	image2 = image2.resize(image1.size)
	psnr = psnrnew(image1,image2)
	print('PSNR:' + str(psnr))
	value = compare_ssim(image1, image2)
	print('SSIM:' + str(value))


if __name__ == '__main__':
    _main()