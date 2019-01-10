# onnxModelRunner
This is test tool for geting test-output-data of onnx models.

## Prerequisites
* Required `python 3.6.x`
* Required [onnx](https://github.com/onnx/onnx) project.

	```
	sudo apt-get install protobuf-compiler libprotoc-dev
	pip3 install onnx
	```

* Required [onnx-models](https://github.com/onnx/models) project.

	```
	pip3 install tensorflow onnx-tf numpy
	```

## Run Tests

```sh
$ npm start
```

## About onnx model
You can add more onnx models into folder `./models`

|  inception_v2  |   mobilenetv2-1.0   |  resnet50v1  |  resnet50v2  |  squeezenet1.1  |
|      :---:     |        :---:        |     :---:    |     :---:    |      :---:      |
|      PASS      |        PASS         |     PASS     |     PASS     |      PASS       |
