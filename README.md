# GRPC-Request

### What is Grpc?
In gRPC, a client application can directly call a method on a server application on a different machine as if it were a local object, making it easier for you to create distributed applications and services. As in many RPC systems, gRPC is based around the idea of defining a service, specifying the methods that can be called remotely with their parameters and return types. On the server side, the server implements this interface and runs a gRPC server to handle client calls. On the client side, the client has a stub (referred to as just a client in some languages) that provides the same methods as the server. [source](https://grpc.io/docs/what-is-grpc/introduction/)


### Demo of streaming
sample requesting result:


### YOLO served model
I used the tensorflow version of YOLO and serve it with Tensorflow-Serving.
This is grpc request to served model and preprosess the output.
#### Installation
```
pip install -r requirements.txt
```

#### How to use
```
docker run -t --rm -p 8500:8500 \
    -v "/path/to/saved_model_yolo_model:/models/yolo" \
    -e MODEL_NAME=yolo \
    tensorflow/serving
    
python grpcrequest2yolo.py
```
