import bz2
import lzma
import time
import zlib
import msgpack
from Resnet_infer import Inference
import json
import sys
import torch
import gzip
import pickle
import lz4.frame
import brotli
import zstd
import snappy

# data = ({"이상민" : 33, "이상민2" : 33, "김재엽" : 90},{"박재형" : 33, "박재형2" : 33, "김재엽" : 90})


infer = Inference()

infer.set_variable()
infer.set_epoch(1)
infer.run()
data = infer.parameter_extract()

torch.save(infer.model, "testmodel(resnet50).pt") # 모델 자체 저장

start_time = time.time()
torch.save(data, "testdata(resnet50).pt") # 파라미터 저장
end_time = time.time()
print("resnet50 테스트")
print("torch.save 수행 시간 : ", round(end_time - start_time, 2), " 초")

# data = torch.load("testdata.pt")
# print(type(data))
# print(data.keys())

# infer.load_parameter(data)
# print(infer.model.state_dict().keys())
# print("성공인듯?")

# print(data)
print("추출된 raw 데이터 타입 : ", type(data))
print("파라미터 사이즈 : ", sys.getsizeof(data))


# msgpack은 tensor 오프젝트 직렬화 불가
# test4 = msgpack.packb(data)
# print("바이너리로 변환한 후 타입(msgpack) : ", type(test4))
# print("바이너리 파라미터 사이즈 : %.2f MB" % (sys.getsizeof(test4) / (1024.0*1024.0)))

# comp_data = gzip.compress(test4)
# print("msgpack 압축 후 바이너리 파라미터 사이즈 : %.2f MB" % (sys.getsizeof(comp_data) / (1024.0*1024.0)))

print("="*10)
# 직렬화 시간 측정
start_time = time.time()
test5 = pickle.dumps(data)
end_time = time.time()
print("pickle 수행 시간 : ", round(end_time - start_time, 2), " 초")
print("바이너리로 변환한 후 타입(pickle) : ", type(test5))
print("바이너리 파라미터 사이즈 : %.2f MB" % (sys.getsizeof(test5) / (1024.0*1024.0)))
print("바이트 = ", sys.getsizeof(test5))

# comp_data = zstd.compress(test5)
# print("zstd 압축 후 바이너리 파라미터 사이즈 : %.2f MB" % (sys.getsizeof(comp_data) / (1024.0*1024.0)))
# print("바이트 = ", sys.getsizeof(comp_data))

# decompressed_data = zstd.decompress(comp_data)
# print("압축 해제 후 바이너리 파라미터 사이즈 : %.2f MB" % (sys.getsizeof(decompressed_data) / (1024.0*1024.0)))
# print("바이트 = ", sys.getsizeof(decompressed_data))

# infer.load_parameter(pickle.loads(decompressed_data))
# infer.run()
# print("성공인듯?")

# 압축 실험
def measure_compression(compression_func, data):
    start_time = time.time()
    compressed_data = compression_func(data)
    end_time = time.time()
    compression_time = round(end_time - start_time, 2)  # 압축 시간을 소수점 둘째 자리까지 반올림
    original_size = sys.getsizeof(data)
    compressed_size = sys.getsizeof(compressed_data)
    compression_ratio = round((1 - compressed_size / original_size) * 100, 2)  # 압축률 계산
    return compression_time, compression_ratio, compressed_size

def format_size(size_in_bytes):
    size_in_mb = size_in_bytes / (1024.0 * 1024.0)
    return size_in_mb, size_in_bytes
# 각 압축 알고리즘의 압축 시간 측정
compressors = {
    'gzip  ': (gzip.compress, (test5,)),
    'zlib  ': (zlib.compress, (test5,)),
    'bz2   ': (bz2.compress, (test5,)),
    'lzma  ': (lzma.compress, (test5,)),
    'lz4   ': (lz4.frame.compress, (test5,)),
    'brotli': (brotli.compress, (test5,)),
    'zstd  ': (zstd.compress, (test5,)),
    'snappy': (snappy.compress, (test5,)),  # snappy 모듈 사용 시
}

for name, (compressor, args) in compressors.items():
    duration, ratio, compressed_size = measure_compression(compressor, *args)
    compressed_size_mb, compressed_size_bytes = format_size(compressed_size)
    print(f"압축 시간 ({name}): {duration} 초, 압축률: {ratio}%, 압축 후 크기: {compressed_size_mb:.2f} MB ({compressed_size_bytes} 바이트)")

# 압축실험


# comp_data2 = gzip.compress(test5)
# print("압축 후 바이너리 파라미터 사이즈(gzip) : %.2f MB" % (sys.getsizeof(comp_data2) / (1024.0*1024.0)))
# print("바이트 = ", sys.getsizeof(comp_data2))

# comp_data3 = zlib.compress(test5)
# print("압축 후 바이너리 파라미터 사이즈(zlib) : %.2f MB" % (sys.getsizeof(comp_data3) / (1024.0*1024.0)))
# print("바이트 = ", sys.getsizeof(comp_data3))

# comp_data4 = bz2.compress(test5)
# print("압축 후 바이너리 파라미터 사이즈(bz2) : %.2f MB" % (sys.getsizeof(comp_data4) / (1024.0*1024.0)))
# print("바이트 = ", sys.getsizeof(comp_data4))

# comp_data5 = lzma.compress(test5)
# print("압축 후 바이너리 파라미터 사이즈(lzma) : %.2f MB" % (sys.getsizeof(comp_data5) / (1024.0*1024.0)))
# print("바이트 = ", sys.getsizeof(comp_data5))

# # lz4를 이용한 압축
# comp_data6 = lz4.frame.compress(test5)
# print("압축 후 바이너리 파라미터 사이즈(lz4) : %.2f MB" % (sys.getsizeof(comp_data6) / (1024.0*1024.0)))
# print("바이트 = ", sys.getsizeof(comp_data6))

# brotli_data = brotli.compress(test5)
# print("압축 후 바이너리 파라미터 사이즈(brotli) : %.2f MB" % (sys.getsizeof(brotli_data) / (1024.0*1024.0)))
# print("바이트 = ", sys.getsizeof(brotli_data))

# decompressed_data = brotli.decompress(brotli_data)
# print("압축 해제 후 바이너리 파라미터 사이즈(brotli) : %.2f MB" % (sys.getsizeof(decompressed_data) / (1024.0*1024.0)))
# print("바이트 = ", sys.getsizeof(decompressed_data))


# infer.load_parameter(pickle.loads(decompressed_data))
# infer.run()
# print("성공인듯?")

# zstd_data = zstd.compress(test5)
# print("압축 후 바이너리 파라미터 사이즈(zstd) : %.2f MB" % (sys.getsizeof(zstd_data) / (1024.0*1024.0)))
# print("바이트 = ", sys.getsizeof(zstd_data))

# comp_data_snappy = snappy.compress(test5)
# print("압축 후 바이너리 파라미터 사이즈(snappy) : %.2f MB" % (sys.getsizeof(comp_data_snappy) / (1024.0*1024.0)))
# print("바이트 = ", sys.getsizeof(comp_data_snappy))

# ==============================


# numpy_data = {key: infer.tensor_to_numpy(value) for key, value in data.items()}
# print("파라미터 넘파이로 변환")
# print("="*10)
# print("넘파이 파라미터 사이즈 : " , sys.getsizeof(numpy_data))
# # print(numpy_data)
# print("넘파이로 변환한 데이터 : ", type(numpy_data))
# print("="*10)

# json_data = json.dumps(numpy_data)
# print("json 파라미터 사이즈 : %.2f MB" % (sys.getsizeof(json_data) / (1024.0*1024.0)))
# print("json으로 변환한 데이터 타입 : ", type(json_data))

# testd2 = msgpack.packb(numpy_data)
# print("msgpack 이용 바이너리 파라미터 사이즈 : %.2f MB" % (sys.getsizeof(testd2) / (1024.0*1024.0)))
# print("넘파이에서 바이너리로 변환한 후 타입 : ", type(testd2))
# print("="*10)

# testd3 = pickle.dumps(numpy_data)
# print("pickle 이용 바이너리 파라미터 사이즈 : %.2f MB" % (sys.getsizeof(testd3) / (1024.0*1024.0)))
# print("넘파이에서 바이너리로 변환한 후 타입 : ", type(testd3))
# print("="*10)

# comp_data = gzip.compress(testd2)
# print("msgpack 압축 후 바이너리 파라미터 사이즈 : %.2f MB" % (sys.getsizeof(comp_data) / (1024.0*1024.0)))
# print("압축 후 데이터 타입 : ", type(comp_data))
# print("="*10)
# decompressed_data = gzip.decompress(comp_data)

# comp_data2 = gzip.compress(testd3)
# print("pickle 압축 후 바이너리 파라미터 사이즈 : %.2f MB" % (sys.getsizeof(comp_data2) / (1024.0*1024.0)))
# print("압축 후 데이터 타입 : ", type(comp_data2))
# print("="*10)

# unpack2 = msgpack.unpackb(decompressed_data)
# print("언팩한 후 파라미터 사이즈 : ", sys.getsizeof(unpack2))
# print("바이너리에서 넘파이로 변환한 후 타입 : ", type(unpack2))

# json_data = json.dumps(numpy_data)
# print("json으로 변환")
# print("json 파라미터 사이즈 : %.2f MB" % (sys.getsizeof(json_data) / (1024.0*1024.0)))
# # print(json_data)

# testd = msgpack.packb(json_data)
# print("바이너리로 변환한 후 타입 : ", type(testd))
# print("바이너리 파라미터 사이즈 : %.2f MB" % (sys.getsizeof(testd) / (1024.0*1024.0)))
# # print(testd)

# # for i in msgpack.unpackb(testd):
# #     print(i)

# unpack = msgpack.unpackb(testd)
# print("size : %.2f MB" % (sys.getsizeof(unpack) / (1024.0*1024.0)))
# print("다시 json : ", type(unpack))
# print("언팩 성공")

### 
# 파라미터 사이즈 :  9344
# 파라미터 넘파이로 변환
# ==========
# 넘파이 파라미터 사이즈 :  3328
# 넘파이로 변환한 데이터 :  <class 'dict'>
# ==========
# json 파라미터 사이즈 : 247.72 MB
# json으로 변환한 데이터 타입 :  <class 'str'>
# msgpack 이용 바이너리 파라미터 사이즈 : 101.06 MB
# 넘파이에서 바이너리로 변환한 후 타입 :  <class 'bytes'>
# ==========
# pickle 이용 바이너리 파라미터 사이즈 : 115.71 MB
# 넘파이에서 바이너리로 변환한 후 타입 :  <class 'bytes'>
# ==========
# msgpack 압축 후 바이너리 파라미터 사이즈 : 49.44 MB
# 압축 후 데이터 타입 :  <class 'bytes'>
# ==========
# pickle 압축 후 바이너리 파라미터 사이즈 : 50.83 MB
# 압축 후 데이터 타입 :  <class 'bytes'>
# ==========
# 결론 : 걍 torch.save 쓰자