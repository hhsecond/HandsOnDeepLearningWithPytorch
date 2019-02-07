import json
import os

import mxnet as mx
from mxnet.io import DataBatch

import time


class MXNetModelService(object):

    def __init__(self):
        self.mxnet_ctx = None
        self.mx_model = None
        self.signature = None
        self.epoch = 0
        self.error = None

    def get_model_files_prefix(self, context):
        return context.manifest["model"]["modelName"]

    def binary_encoder(self, input_num, input_size):
        ret = [int(i) for i in '{0:b}'.format(input_num)]
        return [0] * (input_size - len(ret)) + ret

    def get_readable_output(self, input_num, prediction):
        input_output_map = {
            0: 'FizBuz',
            1: 'Buz',
            2: 'Fiz'}
        if prediction == 3:
            return input_num
        else:
            return input_output_map[prediction]

    def initialize(self, context):
        # todo - check batch size and read it from the message

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        gpu_id = properties.get("gpu_id")
        self._batch_size = properties.get('batch_size')

        signature_file_path = os.path.join(model_dir, "signature.json")
        if not os.path.isfile(signature_file_path):
            raise RuntimeError("Missing signature.json file.")
        # todo - hard coding this value and try what all depends on this fiel
        with open(signature_file_path) as f:
            self.signature = json.load(f)

        model_files_prefix = self.get_model_files_prefix(context)

        data_names = []
        data_shapes = []
        input_data = self.signature["inputs"][0]
        data_name = input_data["data_name"]
        data_shape = input_data["data_shape"]
        # Set batch size
        data_shape[0] = self._batch_size
        # Replace 0 entry in data shape with 1 for binding executor.
        for idx in range(len(data_shape)):
            if data_shape[idx] == 0:
                data_shape[idx] = 1
        data_names.append(data_name)
        data_shapes.append((data_name, tuple(data_shape)))

        checkpoint_prefix = "{}/{}".format(model_dir, model_files_prefix)
        # Load MXNet module
        self.mxnet_ctx = mx.cpu() if gpu_id is None else mx.gpu(gpu_id)
        sym, arg_params, aux_params = mx.model.load_checkpoint(
            checkpoint_prefix, self.epoch)

        # noinspection PyTypeChecker
        self.mx_model = mx.mod.Module(
            symbol=sym, context=self.mxnet_ctx, data_names=data_names, label_names=None)
        self.mx_model.bind(for_training=False, data_shapes=data_shapes)
        self.mx_model.set_params(
            arg_params, aux_params, allow_missing=True, allow_extra=True)
        self.has_initialized = True

    def preprocess(self, batch):
        # todo - assert batch size

        param_name = self.signature['inputs'][0]['data_name']
        data = batch[0].get('body').get(param_name)
        self.input = data + 1
        tensor = mx.nd.array([self.binary_encoder(self.input, input_size=10)])
        return tensor

    def inference(self, model_input):
        if self.error is not None:
            return None

        # Check input shape
        check_input_shape(model_input, self.signature)
        self.mx_model.forward(DataBatch([model_input]))
        model_input = self.mx_model.get_outputs()
        # by pass lazy evaluation get_outputs either returns a list of nd arrays
        # a list of list of NDArray
        for d in model_input:
            if isinstance(d, list):
                for n in model_input:
                    if isinstance(n, mx.ndarray.ndarray.NDArray):
                        n.wait_to_read()
            elif isinstance(d, mx.ndarray.ndarray.NDArray):
                d.wait_to_read()
        return model_input

    def postprocess(self, inference_output):
        if self.error is not None:
            return [self.error] * self._batch_size
        prediction = self.get_readable_output(
            self.input,
            int(inference_output[0].argmax(1).asscalar()))
        out = [{'next_number': prediction}]
        return out

    def handle(self, data, context):
        try:
            if not self.has_initialized:
                self.initialize()
            preprocess_start = time.time()
            data = self.preprocess(data)
            inference_start = time.time()
            data = self.inference(data)
            postprocess_start = time.time()
            data = self.postprocess(data)
            end_time = time.time()

            metrics = context.metrics
            metrics.add_time(
                "PreprocessTime", round((inference_start - preprocess_start) * 1000, 2))
            metrics.add_time(
                "InferenceTime", round((postprocess_start - inference_start) * 1000, 2))
            metrics.add_time(
                "PostprocessTime", round((end_time - postprocess_start) * 1000, 2))
            return data
        except Exception as e:
            request_processor = context.request_processor
            request_processor.report_status(500, "Unknown inference error")
            return [str(e)] * self._batch_size


def check_input_shape(inputs, signature):
    # todo
    pass
