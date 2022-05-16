# project_ds301
# Neural Style Transfer - description 
My project is based on Neural Style Transfer- Mixing Art with Humans - Babies Mixing with Starry Night painting 
Neural style transfer is an optimization technique used to take two images—a content image (baby photo) and a style reference image (Van Gogh's Starry Night)—and blend them together so the output image looks like the content image, but merged in the style of the style reference image.

This is implemented by optimizing the output image to match the content statistics of the content image and the style statistics of the style reference image. These statistics are extracted from the images using a convolutional network.

Cnns are really good at feature extraction of images. Cnns are also used for computer vision. We use a pretrained Cnn and realise the backpropogation mechanics to minimise the objective function. So, for this we minimise the style difference between S (Starry night) and G (generated image) while minimising the content difference between C (baby photo) and G (generated image).This gives us 2 loss functions which we minimise by changing content weights. After which we realise the total variation loss to minimise the difference in the higher frequency components of the images. After which we re run the model and get the desired results. We run the same images on vgg16 model.


# Code structure
1. Define the style and content image
2. Implement the original style transfer algorithm (TensorFlow) T-hub model
Define content and Style (Use a VGG19 and test run it on the image)
3. We use intermediate layers of the model to get the content and style representations of the image. Starting from the network's input layer, the first few layer activations represent low-level features like edges and textures. As you step through the network, the final few layers represent higher-level features—object parts like celestial objects or eyes. In this case, we use the VGG19 network architecture, a pretrained image classification network. These intermediate layers are necessary to define the representation of content and style from the images. This architecture helps an input image to match the corresponding style and content target representations at these intermediate layers.
4. Build the Model- The networks in tf.keras.applications are designed to extract the intermediate layer values using the Keras functional API.
model = Model(inputs, outputs)
This following function builds a VGG19 model that returns a list of intermediate layer outputs
5. Calculate the style- We calculate the gram matrix which takes the outer product 
6. Extract style and conent through our model- that gives out style and content tensors
7. After implementing the model - we run gradient descent and calculate the total variation loss
8. Take into account the total variation loss and change the content weights in higher frequency edge vectors and re-run it in VGG19
9. Repeat the same for VGG16

# Commands
Import and configure modules and create a function to import the images
``` python
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)
``

Visualize the input image by creating a function

``` python
def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)
``

Use Tensorflow Hub
``` python
import tensorflow_hub as hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
tensor_to_image(stylized_image)
``

Define content and style representations by using vgg19 architecture and load the layers
``` python
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

print()
for layer in vgg.layers:
  print(layer.name)
``
Create the model (VGG19)
``` python
style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)
``
Extract style and content
``` python
extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))
``
Variation loss function 
``` python
tf.image.total_variation(image).numpy()
``
Include the loss weight and re-run the model
``` python
import time
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
    print(".", end='', flush=True)
  display.clear_output(wait=True)
  display.display(tensor_to_image(image))
  print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))
``
Repeat the same steps and implement vgg16
``` python
image = tf.Variable(content_image)   
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
    print(".", end='', flush=True)
  display.clear_output(wait=True)
  display.display(tensor_to_image(image))
  print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))
```

# Results VGG19 vs VGG16
Using Tf-Hub model
![tf hub](https://user-images.githubusercontent.com/69249063/168494811-9057d3fc-cdfc-4526-8334-53b12940091a.png)

Using Vgg19
![vgg19 p](https://user-images.githubusercontent.com/69249063/168494817-83699b81-3ec2-4e0f-ac25-f13ec201237c.png)

Using Vgg16

![vgg16 blur](https://user-images.githubusercontent.com/69249063/168496094-4a47b94a-f37c-46b8-9fd6-c6ae585e780a.png)
