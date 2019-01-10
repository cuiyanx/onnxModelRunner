import onnx
import os
import glob
import csv
from onnx import numpy_helper
from onnx_tf.backend import prepare

# Load inputs
def load_input_data(input_path):
	inputs = []
	inputs_num = len(glob.glob(os.path.join(input_path , "input_*.pb")))

	for i in range(inputs_num):
		input_file = os.path.join(input_path , "input_{}.pb".format(i))
		tensor = onnx.TensorProto()

		with open(input_file, "rb") as f:
			tensor.ParseFromString(f.read())

		inputs.append(numpy_helper.to_array(tensor))

	return inputs

# Load reference outputs
def load_output_data(output_path):
	outputs = []
	outputs_num = len(glob.glob(os.path.join(output_path , "output_*.pb")))

	for i in range(outputs_num):
		output_file = os.path.join(output_path , "output_{}.pb".format(i))
		tensor = onnx.TensorProto()

		with open(output_file, "rb") as f:
			tensor.ParseFromString(f.read())

		outputs.append(numpy_helper.to_array(tensor))

	return outputs

# Run the model on the backend
def get_output_data(onnx_model, input_data):
	output = list(prepare(onnx_model).run(input_data))

	return output

def get_onnx_model_path(model_path):
	models = glob.glob(os.path.join(model_path , "*.onnx"))

	if len(models) == 0:
		return None
	else:
		return models[0]

def get_test_path(model_path):
	tests_path = glob.glob(os.path.join(model_path , "test_data_set_*"))

	for test_path in tests_path:
		if not os.path.isdir(test_path):
			tests_path.remove(test_path)

	return tests_path


if __name__ == '__main__':
	output_path = "./output"
	models_path = "./models"
	models_dict = dict()
	csv_data = []
	headers_all = ["model_name", "model_path", "test_name", "test_path", "output_data"]

	if os.path.exists(output_path):
		cmd = "rm -r " + output_path
		os.system(cmd)

	if not os.path.exists(output_path):
		os.makedirs(output_path)

	if os.path.exists(models_path):
		names = os.listdir(models_path)

		for name in names:
			dir_or_file = os.path.join(models_path, name)

			if os.path.isdir(dir_or_file):
				models_dict[name] = dir_or_file

	models_list = models_dict.keys()

	for model_name in models_list:
		model_path = models_dict.get(model_name)

		onnx_model_path = get_onnx_model_path(model_path)
		onnx_model_name = onnx_model_path[len(model_path) + 1:]
		onnx_model = onnx.load(onnx_model_path)

		tests_path = get_test_path(model_path)

		for test_data_dir in tests_path:
			test_data_name = test_data_dir[len(model_path) + 1:]
			test_data_input = load_input_data(test_data_dir)
			test_data_output_ref = load_output_data(test_data_dir)
			test_data_output = get_output_data(onnx_model, test_data_input[0])

			csv_dict = dict()
			csv_dict["model_name"] = model_name
			csv_dict["model_path"] = model_path
			csv_dict["test_name"] = test_data_name
			csv_dict["test_path"] = test_data_dir
			csv_dict["output_data"] = test_data_output[0]
			csv_data.append(csv_dict)

	with open(os.path.join(output_path , "output_data.csv"), "w", newline="") as f:
		writer = csv.DictWriter(f, headers_all)
		writer.writeheader()
		writer.writerows(csv_data)
