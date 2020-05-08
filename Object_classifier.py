import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D,Dropout
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam

# Class to generate Model Classifier
class Classifier():
    
	def __init__(self):
		pass

	# Image reprocessing
	def prepare_image(self,file):
    	img_path = 'Image'
    	img = image.load_img(img_path + file, target_size=(224, 224))
    	img_array = image.img_to_array(img)
    	img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    	return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

    # Generate model
	def generate_model(self):
		# get MobileNet application from Keras
		mobile = keras.applications.mobilenet.MobileNet()
		# Import the base MobileNet model and ignore the last 1000 neuron layers
		base_model=MobileNet(weights='imagenet',include_top=False)

		# set layers if the model
		x=base_model.output
		# Create pooling 
		x=GlobalAveragePooling2D()(x)
		# Add dense layer to fully connect all the layer and learn complex functions
		x=Dense(1024,activation='relu')(x) 
		# Dense layer 2 to improve the classifiication accuracy
		x=Dense(1024,activation='relu')(x) 
		# Add Dropout layer to reduce overfitting
		x=Dropout(0.2)(x)
		# Dense layer 3 to learn classes more accurately after dropping random neuron layers
		x=Dense(512,activation='relu')(x) 
		# Softmax layer to convert output in range 0 to 1
		preds=Dense(2,activation='softmax')(x) 

		# Create model with all the mentoned layers
		model=Model(inputs=base_model.input,outputs=preds)

		# Select layers to train
		for layer in model.layers:
    		layer.trainable=False
		# or if we want to set the first 20 layers of the network to be non-trainable
		for layer in model.layers[:20]:
		    layer.trainable=False
		for layer in model.layers[20:]:
		    layer.trainable=True

		# Generate train test and validation dataset
		train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)
		# Training dataset generator
		train_generator=train_datagen.flow_from_directory('Image',
        											target_size=(224,224),
        											color_mode='rgb',
        											batch_size=32,
        											subset = 'training',
        											class_mode='categorical',
        											shuffle=True)
		# Validation dataset genartor
		test_generator=train_datagen.flow_from_directory('Image',
                                                  target_size=(224,224),
                                                  color_mode='rgb',
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  subset = 'validation',
                                                  shuffle=True)

		# Compile model with categorical cross entropy as loss function
		# Adam optimizer is user for better performance - sgd was not giving any good accuracy
		model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
		# Training step size
		step_size_train=train_generator.n//train_generator.batch_size
		# Train model for specified epochs
		model.fit_generator(generator=train_generator, validation_data = test_generator
		                   steps_per_epoch=step_size_train,
		                   epochs=10)
		# Save the model
		model.save("CarType_Classifer_Mobilenet_TF115.h5")