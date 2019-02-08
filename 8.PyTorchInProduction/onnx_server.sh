cd FizBuz_with_ONNX
model-archiver --model-name fizbuz_package --model-path fizbuz_package --handler fizbuz_service -f
cd ..
mxnet-model-server --start --model-store FizBuz_with_ONNX --models fizbuz_package.mar

# curl -v -X PUT "http://localhost:8081/models/fizbuz_package?max_worker=1"
# curl  "http://localhost:8081/models/fizbuz_package"
# curl -X POST http://127.0.0.1:8080/predictions/fizbuz_package -H "Content-Type: application/json" -d '{"input.1": 14}'
