# Copyright 2017-2018 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test", "plaidml_proto_library")

plaidml_proto_library(
    name = "proto",
    srcs = [
        "cm.proto",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/context:proto",
        "//tile/proto:hal",
    ],
)

plaidml_cc_library(
    name = "cm",
    srcs = [
    	"buffer.cc",
        "buffer.h",
        "mem_arena.cc",
        "mem_arena.h",
        "mem_buffer.cc",
        "mem_buffer.h",
        "compiler.cc",
        "compiler.h",
        "compute_kernel.cc",
        "compute_kernel.h",
        "device.cc",
        "device.h",
        "device_set.cc",
        "device_set.h",
        "device_state.cc",
        "device_state.h",
        "driver.cc",
        "driver.h",
        "event.cc",
        "event.h",
        "executable.cc",
        "executable.h",
        "executor.cc",
        "executor.h",
        "host_memory.cc",
        "host_memory.h",
        "result.cc",
        "result.h", 
        "library.cc",
        "library.h", 
        "info.cc",   
        "info.h",
        "err.cc", 
        "err.h", 
        "zero_kernel.cc",
        "zero_kernel.h",
        "opt.h",
        "opt.cc",
        "emitcm.cc",
        "emitcm.h",
        "kernel.h",
        "cm.cc",
        "shared_memory.cc",
    ],
    hdrs = [
    	"runtime.h",
    ],
    copts = [
  		'-Iexternal/cmheaders',
	],
	linkopts=[
		'-rdynamic /home/yangleiz/Desktop/MDF_internal/runtime/lib/x64/libigfxcmrt.so',
	],
    visibility = ["//visibility:public"],
    deps = [
        ":proto_cc",
        "//base/util",
        "@boost//:filesystem",
        "@boost//:regex",
		#"@cm//:cm_headers",
        "@cmheaders//:headers",
        "//tile/base",
        "//tile/base:hal",
        "//tile/hal/util:selector",
        "//tile/lang",
        "//tile/proto:proto_cc",
        "//tile/proto:support",
    ],
    alwayslink = 1,
)

plaidml_cc_test(
    name = "platform_test",
    srcs = ["platform_test.cc"],
    deps = [
        ":cm",
        "//tile/base:platform_test",
        "//tile/platform/local_machine",
    ],
)
