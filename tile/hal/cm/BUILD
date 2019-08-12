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
    	"cm_buffer.cc",
        "cm_buffer.h",
        "cm_mem_arena.cc",
        "cm_mem_arena.h",
        "cm_mem_buffer.cc",
        "cm_mem_buffer.h",
        "cm_compiler.cc",
        "cm_compiler.h",
        "cm_compute_kernel.cc",
        "cm_compute_kernel.h",
        "cm_device.cc",
        "cm_device.h",
        "cm_device_set.cc",
        "cm_device_set.h",
        "cm_device_state.cc",
        "cm_device_state.h",
        "cm_driver.cc",
        "cm_driver.h",
        "cm_event.cc",
        "cm_event.h",
        "cm_executable.cc",
        "cm_executable.h",
        "cm_executor.cc",
        "cm_executor.h",
        "cm_host_memory.cc",
        "cm_host_memory.h",
        "cm_result.cc",
        "cm_result.h", 
        "cm_library.cc",
        "cm_library.h", 
        "cm_info.cc",   
        "cm_info.h",
        "cm_err.cc", 
        "cm_err.h", 
        "cm_zero_kernel.cc",
        "cm_zero_kernel.h",
        "cm_opt.h",
        "cm_opt.cc",
        "emitcm.cc",
        "emitcm.h",
        #"emitcm_dbg.cc",
        #"emitcm_dbg.h",
        "kernel.h",
        "cm.cc",
        "cm_shared_memory.cc",
    ],
    hdrs = [
    	"cm_runtime.h",
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
