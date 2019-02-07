cd FizBuz_with_ONNX
model-archiver --model-name fizbuz_package --model-path fizbuz_package --handler fizbuz_service -f
cd ..
mxnet-model-server --start --model-store FizBuz_with_ONNX --models fizbuz_package.mar