import pandas as pd

# Tensorflow/Keras Imports
import tensorflow 
from tensorflow import keras 
#from keras import callbacks
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers

# AzureML Imports
from azureml.core import Run, Dataset, Workspace

import argparse
import warnings
warnings.filterwarnings("ignore")
num_classes = 200
RGB_channels = 3

# Authentication issue...
from azureml.core.authentication import InteractiveLoginAuthentication

interactive_auth = InteractiveLoginAuthentication(tenant_id="<YOUR TENANT ID HERE>")
""" 
Get default workspace. Interactive auth was having trouble for accounts that have more than one subscription.
E.g. for us that have a UW one and the Microsoft Operationalization one. It would default to UW and not authenticate 
the compute instance to run the pipeline. Also don't have access to IAM controlls so we cannot add a managed identity
to the compute instances
"""
ws = Workspace.get(
    subscription_id="<YOUR SUBSCRIPTION ID HERE>",
    resource_group="<RESOURCE GROUP HERE>",
    name="<WORKSPACE NAME>",
    auth=interactive_auth
)

import pickle, json
import joblib, os
run = Run.get_context()


# Download the data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default="./data",
        help="Path to ImageNet Dataset"
    )
    parser.add_argument(
        '--num_epochs', 
        type=int,
        default=15,
        help="Number of epochs to train"
    )
    parser.add_argument(
        '--steps_per_epoch',
        type=int,
        default=200,
        help="Number of steps per epoch"
    )
    parser.add_argument(
        '--target_img_size', 
        type=int, 
        default=224, 
        help="Dimensions of image. Width and height will be set to this value."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size. Make sure there's enough data for (steps per epoch) * (number of epochs)"
    )
    parser.add_argument(
        '--save_dest',
        type=str,
        default="./outputs",
        help="Directory to save model to."
    )
    parser.add_argument(
        "--run_eval",
        type=bool,
        default=False,
        help="Set to True (boolean) to run on evaluation split after training."
    )
    """
    parser.add_argument(
        "--conv_to_onnx",
        type=bool,
        default=False,
        help="Set to True (boolean) to convert model to ONNX."
    )
    """
    args = parser.parse_args()

    """
    # Download dataset
    print(f"Downloading Tiny Imagenet dataset to {args.data_path}...")
    dataset = Dataset.get_by_name(workspace=ws, name='Tiny ImageNet')
    dataset.download(target_path=args.data_path, overwrite=False)
    print("Done downloading.\n")
    """
    #os.makedirs(f"/{args.save_dest}", exist_ok=True)
    inception_base = InceptionV3(weights='imagenet', include_top=False, 
        input_shape=(None, None, 3))
    x = inception_base.output
    x = GlobalAveragePooling2D()(x) # Reduce overall number of parameters to minimize overfitting
    x = Dense(1024, activation='relu')(x)
    # Output layer. 200 classes
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inception_base.input, outputs=predictions)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in inception_base.layers:
        layer.trainable = False
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    #Get validation data
    val_data = pd.read_csv(f"{args.data_path}/val/val_annotations.txt", sep="\t",
        header=None, names=['file', 'class', '1', '2', '3', '4'])
    val_data.drop(['1', '2', '3', '4'], axis=1, inplace=True)
    
    # Normalize each RGB value. RGB values = 0 to 255 (inclusive). 
    # Converts to a floating point between 0.0 and 1.0 inclusive.
    target_img_size = (args.target_img_size, args.target_img_size)
    train_datagen, valid_datagen = ImageDataGenerator(rescale=1.0/255), ImageDataGenerator(rescale=1.0/255)
    
    seed=0

    # Create data generators
    train_data_generator = train_datagen.flow_from_directory(f"{args.data_path}/train/",
                                target_size=target_img_size, color_mode="rgb", 
                                batch_size=args.batch_size, class_mode="categorical", 
                                shuffle=True, seed=seed)
    validation_data_generator = valid_datagen.flow_from_dataframe(val_data, 
                                    directory = f"{args.data_path}/val/images", 
                                    x_col="file", y_col="class",
                                    target_size=target_img_size, color_mode="rgb",
                                    class_mode ="categorical", batch_size=args.batch_size,
                                    shuffle=True, seed=seed)

    checkpoint = ModelCheckpoint(
                    filepath=f"{args.save_dest}/checkpoints/",
                    save_weights_only=True,
                    verbose=1
                )
    # Fit the model
    history = model.fit(train_data_generator, epochs=args.num_epochs, steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.steps_per_epoch,
        validation_data=validation_data_generator, verbose=1,
        callbacks=[])
    print("Finished first round of training.")

    run_json = {
        "num_epochs": args.num_epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "target_img_size": args.target_img_size
    }
    # If directory to save to doesn't exist, make it first.
    if not os.path.isdir(args.save_dest):
        os.mkdir(args.save_dest)
    model.save(f"{args.save_dest}/inceptionv3_imagenet")
    joblib.dump(value=run_json, filename=f"{args.save_dest}/run_params.pkl")
    joblib.dump(value=history.history, filename=f"{args.save_dest}/history.pkl")
    # TODO: Check for if we should run on evaluation set and convert to ONNX if needed. Save to same place.
    if args.run_eval:
        print("TODO: Going to run evaluation on eval split!")


