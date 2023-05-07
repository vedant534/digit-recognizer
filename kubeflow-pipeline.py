import kfp
from kfp import dsl
import kfp.components as components

from typing import NamedTuple
def get_data_batch() -> NamedTuple('Outputs', [('datapoints_training', float),('datapoints_test', float),('dataset_version', str)]):
    """
    Function to get dataset and load it to minio bucket
    """
    print("getting data")
    from tensorflow import keras
    from minio import Minio
    import numpy as np
    import json

    minio_client = Minio(
        "100.65.11.110:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )
    minio_bucket = "mlpipeline"
    
    minio_client.fget_object(minio_bucket,"mnist.npz","/tmp/mnist.npz")
    
    def load_data():
        with np.load("/tmp/mnist.npz", allow_pickle=True) as f:
            x_train, y_train = f["x_train"], f["y_train"]
            x_test, y_test = f["x_test"], f["y_test"]

        return (x_train, y_train), (x_test, y_test)
    
    # Get MNIST data directly from library
    (x_train, y_train), (x_test, y_test) = load_data()

    # save to numpy file, store in Minio
    np.save("/tmp/x_train.npy",x_train)
    minio_client.fput_object(minio_bucket,"x_train","/tmp/x_train.npy")

    np.save("/tmp/y_train.npy",y_train)
    minio_client.fput_object(minio_bucket,"y_train","/tmp/y_train.npy")

    np.save("/tmp/x_test.npy",x_test)
    minio_client.fput_object(minio_bucket,"x_test","/tmp/x_test.npy")

    np.save("/tmp/y_test.npy",y_test)
    minio_client.fput_object(minio_bucket,"y_test","/tmp/y_test.npy")
    
    dataset_version = "1.0"
    
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    from collections import namedtuple
    divmod_output = namedtuple('Outputs', ['datapoints_training', 'datapoints_test', 'dataset_version'])
    return [float(x_train.shape[0]),float(x_test.shape[0]),dataset_version]
    
def get_latest_data():
    """
    Dummy functions for showcasing
    """
    print("Adding latest data")
    
    
def reshape_data():
    """
    Reshape the data for model building
    """
    print("reshaping data")
    
    from minio import Minio
    import numpy as np

    minio_client = Minio(
        "100.65.11.110:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )
    minio_bucket = "mlpipeline"
    
    # load data from minio
    minio_client.fget_object(minio_bucket,"x_train","/tmp/x_train.npy")
    x_train = np.load("/tmp/x_train.npy")
    
    minio_client.fget_object(minio_bucket,"x_test","/tmp/x_test.npy")
    x_test = np.load("/tmp/x_test.npy")
    
    # reshaping the data
    # reshaping pixels in a 28x28px image with greyscale, canal = 1. This is needed for the Keras API
    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)

    # normalizing the data
    # each pixel has a value between 0-255. Here we divide by 255, to get values from 0-1
    x_train = x_train / 255
    x_test = x_test / 255
    
    # save data from minio
    np.save("/tmp/x_train.npy",x_train)
    minio_client.fput_object(minio_bucket,"x_train","/tmp/x_train.npy")
    
    np.save("/tmp/x_test.npy",x_test)
    minio_client.fput_object(minio_bucket,"x_test","/tmp/x_test.npy")

def model_building(
    no_epochs:int = 1,
    optimizer: str = "adam"
) -> NamedTuple('Output', [('mlpipeline_ui_metadata', 'UI_metadata'),('mlpipeline_metrics', 'Metrics')]):
    """
    Build the model with Keras API
    Export model parameters
    """
    from tensorflow import keras
    import tensorflow as tf
    from minio import Minio
    import numpy as np
    import pandas as pd
    import json
    
    minio_client = Minio(
        "100.65.11.110:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )
    minio_bucket = "mlpipeline"
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(keras.layers.MaxPool2D(2, 2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))

    model.add(keras.layers.Dense(32, activation='relu'))

    model.add(keras.layers.Dense(10, activation='softmax')) #output are 10 classes, numbers from 0-9

    #show model summary - how it looks
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    metric_model_summary = "\n".join(stringlist)
    
    #compile the model - we want to have a binary outcome
    model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])
    
    minio_client.fget_object(minio_bucket,"x_train","/tmp/x_train.npy")
    x_train = np.load("/tmp/x_train.npy")
    
    minio_client.fget_object(minio_bucket,"y_train","/tmp/y_train.npy")
    y_train = np.load("/tmp/y_train.npy")
    
    #fit the model and return the history while training
    history = model.fit(
      x=x_train,
      y=y_train,
      epochs=no_epochs,
      batch_size=20,
    )
    
    minio_client.fget_object(minio_bucket,"x_test","/tmp/x_test.npy")
    x_test = np.load("/tmp/x_test.npy")
    
    minio_client.fget_object(minio_bucket,"y_test","/tmp/y_test.npy")
    y_test = np.load("/tmp/y_test.npy")
    

    # Test the model against the test dataset
    # Returns the loss value & metrics values for the model in test mode.
    model_loss, model_accuracy = model.evaluate(x=x_test,y=y_test)
    
    # Confusion Matrix

    # Generates output predictions for the input samples.
    test_predictions = model.predict(x=x_test)

    # Returns the indices of the maximum values along an axis.
    test_predictions = np.argmax(test_predictions,axis=1) # the prediction outputs 10 values, we take the index number of the highest value, which is the prediction of the model

    # generate confusion matrix
    confusion_matrix = tf.math.confusion_matrix(labels=y_test,predictions=test_predictions)
    confusion_matrix = confusion_matrix.numpy()
    vocab = list(np.unique(y_test))
    data = []
    for target_index, target_row in enumerate(confusion_matrix):
        for predicted_index, count in enumerate(target_row):
            data.append((vocab[target_index], vocab[predicted_index], count))

    df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
    cm_csv = df_cm.to_csv(header=False, index=False)
    
    metadata = {
        "outputs": [
            {
                "type": "confusion_matrix",
                "format": "csv",
                "schema": [
                    {'name': 'target', 'type': 'CATEGORY'},
                    {'name': 'predicted', 'type': 'CATEGORY'},
                    {'name': 'count', 'type': 'NUMBER'},
                  ],
                "target_col" : "actual",
                "predicted_col" : "predicted",
                "source": cm_csv,
                "storage": "inline",
                "labels": [0,1,2,3,4,5,6,7,8,9]
            },
            {
                'storage': 'inline',
                'source': '''# Model Overview
## Model Summary

```
{}
```

## Model Performance

**Accuracy**: {}
**Loss**: {}

'''.format(metric_model_summary,model_accuracy,model_loss),
                'type': 'markdown',
            }
        ]
    }
    
    metrics = {
      'metrics': [{
          'name': 'model_accuracy',
          'numberValue':  float(model_accuracy),
          'format' : "PERCENTAGE"
        },{
          'name': 'model_loss',
          'numberValue':  float(model_loss),
          'format' : "PERCENTAGE"
        }]}
    
    ### Save model to minIO
    
    keras.models.save_model(model,"/tmp/detect-digits")
    
    from minio import Minio
    import os

    minio_client = Minio(
            "100.65.11.110:9000",
            access_key="minio",
            secret_key="minio123",
            secure=False
        )
    minio_bucket = "mlpipeline"


    import glob

    def upload_local_directory_to_minio(local_path, bucket_name, minio_path):
        assert os.path.isdir(local_path)

        for local_file in glob.glob(local_path + '/**'):
            local_file = local_file.replace(os.sep, "/") # Replace \ with / on Windows
            if not os.path.isfile(local_file):
                upload_local_directory_to_minio(
                    local_file, bucket_name, minio_path + "/" + os.path.basename(local_file))
            else:
                remote_path = os.path.join(
                    minio_path, local_file[1 + len(local_path):])
                remote_path = remote_path.replace(
                    os.sep, "/")  # Replace \ with / on Windows
                minio_client.fput_object(bucket_name, remote_path, local_file)

    upload_local_directory_to_minio("/tmp/detect-digits",minio_bucket,"models/detect-digits/1/") # 1 for version 1
    
    print("Saved model to minIO")
    
    from collections import namedtuple
    output = namedtuple('output', ['mlpipeline_ui_metadata', 'mlpipeline_metrics'])
    return output(json.dumps(metadata),json.dumps(metrics))

def model_serving():
    """
    Create kserve instance
    """
    from kubernetes import client 
    from kserve import KServeClient
    from kserve import constants
    from kserve import utils
    from kserve import V1beta1InferenceService
    from kserve import V1beta1InferenceServiceSpec
    from kserve import V1beta1PredictorSpec
    from kserve import V1beta1TFServingSpec
    from datetime import datetime

    namespace = utils.get_default_target_namespace()

    now = datetime.now()
    v = now.strftime("%Y-%m-%d--%H-%M-%S")

    name='digits-recognizer-{}'.format(v)
    kserve_version='v1beta1'
    api_version = constants.KSERVE_GROUP + '/' + kserve_version

    isvc = V1beta1InferenceService(api_version=api_version,
                                   kind=constants.KSERVE_KIND,
                                   metadata=client.V1ObjectMeta(
                                       name=name, namespace=namespace, annotations={'sidecar.istio.io/inject':'false'}),
                                   spec=V1beta1InferenceServiceSpec(
                                   predictor=V1beta1PredictorSpec(
                                       service_account_name="sa-minio-kserve",
                                       tensorflow=(V1beta1TFServingSpec(
                                           storage_uri="s3://mlpipeline/models/detect-digits/"))))
    )

    KServe = KServeClient()
    KServe.create(isvc)

comp_get_data_batch = components.create_component_from_func(get_data_batch,base_image="gcr.io/arrikto/jupyter-kale-gpu-tf-py38:release-2.0-l0-release-2.0-39-gc3abc739f")
comp_get_latest_data = components.create_component_from_func(get_latest_data,base_image="gcr.io/arrikto/jupyter-kale-gpu-tf-py38:release-2.0-l0-release-2.0-39-gc3abc739f")
comp_reshape_data = components.create_component_from_func(reshape_data,base_image="gcr.io/arrikto/jupyter-kale-gpu-tf-py38:release-2.0-l0-release-2.0-39-gc3abc739f")
comp_model_building = components.create_component_from_func(model_building,base_image="gcr.io/arrikto/jupyter-kale-gpu-tf-py38:release-2.0-l0-release-2.0-39-gc3abc739f")
comp_model_serving = components.create_component_from_func(model_serving,base_image="gcr.io/arrikto/jupyter-kale-gpu-tf-py38:release-2.0-l0-release-2.0-39-gc3abc739f"
                                                           )


@dsl.pipeline(
    name='digits-recognizer-pipeline',
    description='Detect digits'
)
def output_test(no_epochs,optimizer):
    step1_1 = comp_get_data_batch()
    step1_2 = comp_get_latest_data()
    
    step2 = comp_reshape_data()
    step2.after(step1_1)
    step2.after(step1_2)
    
    step3 = comp_model_building(no_epochs,optimizer)
    step3.after(step2)
    
    step4 = comp_model_serving()
    step4.after(step3)


if __name__ == "__main__":
    client = kfp.Client()

    arguments = {
        "no_epochs" : 1,
        "optimizer": "adam"
    }

    run_directly = 1
    
    if (run_directly == 1):
        client.create_run_from_pipeline_func(output_test,arguments=arguments,experiment_name="test")
    else:
        kfp.compiler.Compiler().compile(pipeline_func=output_test,package_path='output_test.yaml')
        client.upload_pipeline_version(pipeline_package_path='output_test.yaml',pipeline_version_name="0.4",pipeline_name="pipeline test",description="just for testing")
