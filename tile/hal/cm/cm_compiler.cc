// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/cm_compiler.h"

#include <stdlib.h>

#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

#include "base/util/callback_map.h"
#include "base/util/compat.h"
#include "base/util/env.h"
#include "base/util/file.h"
#include "base/util/logging.h"
#include "base/util/uuid.h"
#include "tile/hal/cm/cm_library.h"
#include "tile/hal/cm/cm_opt.h"
#include "tile/hal/cm/cm_runtime.h"
#include "tile/hal/cm/emitcm.h"
#include "tile/lang/semprinter.h"

namespace fs = boost::filesystem;

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

int cmCompiler::knum = 1;

struct cmBuildState;

cmCompiler::cmCompiler(std::shared_ptr<cmDeviceState> device_state) : device_state_{device_state} {}

class cm_Build {
 public:
  cm_Build(context::Activity activity, std::shared_ptr<cmDeviceState> device_state,
           const std::map<std::string, CmProgram*>& program,
           const std::map<std::string, std::shared_ptr<Emit>>& emit_map,
           const std::vector<lang::KernelInfo>& kernel_info, std::vector<context::proto::ActivityID> kernel_ids);

  boost::future<std::unique_ptr<hal::Library>> Start();
  std::unique_ptr<cmLibrary>& library() { return library_; }
  std::shared_ptr<cmDeviceState> device_state() { return device_state_; }

 private:
  void OnError(const std::string& current);
  context::Activity activity_;
  std::shared_ptr<cmDeviceState> device_state_;
  std::unique_ptr<cmLibrary> library_;
  boost::promise<std::unique_ptr<hal::Library>> prom_;
};

struct cmBuildState {
  cmBuildState(cm_Build* b, const std::string& c) : build(b), current(c) {}
  cm_Build* build;
  std::string current;
};

cm_Build::cm_Build(context::Activity activity, std::shared_ptr<cmDeviceState> device_state,
                   const std::map<std::string, CmProgram*>& program_map,
                   const std::map<std::string, std::shared_ptr<Emit>>& emit_map,
                   const std::vector<lang::KernelInfo>& kernel_info, std::vector<context::proto::ActivityID> kernel_ids)
    : activity_{std::move(activity)},
      device_state_{device_state},
      library_{std::make_unique<cmLibrary>(device_state, std::move(program_map), emit_map, kernel_info,
                                           std::move(kernel_ids))} {}

boost::future<std::unique_ptr<hal::Library>> cm_Build::Start() {
  auto result = prom_.get_future();
  prom_.set_value(std::move(library_));
  return result;
}

CmProgram* LoadProgram(CmDevice* pCmDev, const char* code) {
  FILE* pISA = fopen(code, "rb");

  fseek(pISA, 0, SEEK_END);
  int codeSize = ftell(pISA);
  rewind(pISA);

  auto pCommonISACode = malloc(codeSize);

  auto check_err = fread(pCommonISACode, 1, codeSize, pISA);
  if (check_err) {
    check_err = 0;
  }
  fclose(pISA);

  CmProgram* program = NULL;
  pCmDev->LoadProgram(pCommonISACode, codeSize, program);
  free(pCommonISACode);

  return program;
}

std::string kernel_header =  // NOLINT
    R"***(
#include <cm/cm.h>
#include <cm/cmtl.h>

#define _E  2.71828182845904
#define _ln2 0.69314718055995

#define _EXP(_N)	cm_pow(_E,_N,0)
#define _POW(_N,_M)	_pow(_N,_M)
#define _SQRT(_N)	cm_sqrt(_N,0)
#define _LOG(_N)	cm_log(_N,0)*_ln2
#define _SIN(_N)	cm_sin(_N,0)
#define _COS(_N)	cm_cos(_N,0)
#define _TANH(_N)	(1-2/(_EXP(2 * _N)+1))
#define _ROUND(_N)	cm_rndd<float,4>(_N+0.5,0)

#define SCHAR_MIN	(-128)
#define SCHAR_MAX	127
#define SHRT_MIN	(-32768)
#define SHRT_MAX	32767
#define INT_MIN		(-2147483648)
#define INT_MAX		2147483647
#define LONG_MIN	(-9223372036854775808)
#define LONG_MAX	9223372036854775807

#define UCHAR_MAX	255
#define USHRT_MAX	65535
#define UINT_MAX	4294967295
#define ULONG_MAX	18446744073709551615
 
#define FLT_MAX		3.402823e+38		
#define DBL_MAX		1.79769e+308

_GENX_ int _mod(int a, int b)              
{
	return a - ( a / b ) * b;
}


_GENX_ float _pow(float a, long b)              
{
	float r = cm_pow(a, b);
	if(b % 2 == 1 ){
		if(a < 0) r = -r;
	}
	return r;
}

template <typename T, int N>
_GENX_ vector<T,N> _pow(vector_ref<T,N> a, long b)              
{
	vector<T,N> r = cm_pow(a, b);
	if(b % 2 == 1){
		for(int i = 0; i < N; i++){
			if(a(i) < 0) r(i) = -r(i);
		}
	}
	return r;
}

extern "C" _GENX_ int _cmamp(SurfaceIndex idx, int offset, int b, int c)              
{
	vector<int, 4> temp;
	read(idx, offset, temp);
	int a = temp(_mod(offset/sizeof(int), 4));
	if(a < b) return b;
	if(a > c) return c;
	return a;
}

extern "C" _GENX_ uint cast_float_to_uint(float f)              
{
	float* fp = &f;
	uint* uintp = (uint *) fp;
	return *uintp;
}

extern "C" _GENX_ void write_single_atomic_uint(SurfaceIndex index, int offset_bytes, uint ui)              
{
	int offset=offset_bytes/sizeof(uint);
	vector<uint, 4> temp1;
	read(index, sizeof(uint) * (offset), temp1);

	vector<uint, 4> temp0=temp1;
	temp0(offset%4)=ui;

	uint aligned_offset=(uint)(offset-offset%4);
	vector<uint, 4> u;
	for(int i = 0; i < 4; i++){
		u(i) = aligned_offset+i;
	}

	write_atomic<ATOMIC_CMPXCHG,uint>(index, u, temp0, temp1);
}


extern "C" _GENX_ void write_single_atomic_long(SurfaceIndex index, int offset_bytes, long f)              
{
	int offset=offset_bytes/sizeof(uint);
	vector<uint, 4> temp1;
	read(index, sizeof(uint) * (offset), temp1);

	vector<uint, 4> temp0 = temp1;
	temp0(offset%4) = (uint)f;
	
	if(sizeof(long) == 8){
		temp0(offset%4 + 1)=(uint)(f >> 32);
	}

	uint aligned_offset=(uint)(offset-offset%4);
	vector<uint, 4> u;
	for(int i = 0; i < 4; i++){
		u(i) = aligned_offset+i;
	}

	write_atomic<ATOMIC_CMPXCHG,uint>(index, u, temp0, temp1);
}

extern "C" _GENX_ void write_single_atomic_float(SurfaceIndex index, int offset_bytes, float f)              
{
	int offset=offset_bytes/sizeof(float);

	vector<float, 4> original;
	read(index, sizeof(float) * (offset), original);

	vector<uint, 4> temp1 = 0;
	temp1(offset % 4) = cast_float_to_uint(original(offset%4));

	vector<uint, 4> temp0 = temp1;
	temp0(offset % 4) = cast_float_to_uint(f);

	uint aligned_offset = (uint)(offset-offset % 4);

	vector<uint, 4> u;
	for(int i = 0; i < 4; i++){
		u(i) = aligned_offset+i;
	}

	write_atomic<ATOMIC_CMPXCHG,uint>(index, u, temp0, temp1);
}

template <typename T, int N>
_GENX_ void _write(SurfaceIndex suf, int offset_bytes, vector<T,N> v)              
{
	int offset=offset_bytes/sizeof(T);
	if(_mod(offset,16/sizeof(T))==0){
		write(suf,sizeof(T)*offset,v);
	}
	else{
		for(int i=0;i<N;i++){
			write_single_atomic_float(suf, sizeof(T)*(offset+i), v(i)); 	
		}
	}

}
template <typename T, int N>
_GENX_ void _read(SurfaceIndex suf, int offset_bytes, vector_ref<T,N> v)              
{
	int offset=offset_bytes/sizeof(T);
	if(_mod(offset,16/sizeof(T))==0){
		read(suf,sizeof(T)*offset,v);
	}
	else{
		vector<T,2*N> v2=0;
		read(suf,sizeof(T)*offset/16*16,v2);
		for(int i=0;i<N;i++){
			v(i)=v2(i+_mod(offset,16/sizeof(T)));	
		}
	}

}

template <typename T, int N>
_GENX_ void _read(SurfaceIndex suf, int offset, vector_ref<uint,N> element_offset, vector_ref<T,N> v)              
{
	read(suf, offset, element_offset, v);
}

_GENX_ void write_single_char(SurfaceIndex index, int offset_bytes, char c)              
{
	vector<char, 16> temp;
	read(index, offset_bytes/16*16, temp);
	temp(offset_bytes%16)=c;
	write(index, offset_bytes/16*16, temp);
}

_GENX_ void write_single_short(SurfaceIndex index, int offset_bytes, short s)              
{
	vector<short, 8> temp;
	read(index, offset_bytes/16*16, temp);
	temp(offset_bytes/sizeof(short)%8)=s;
	write(index, offset_bytes/16*16, temp);
}

_GENX_ void write_single_half(SurfaceIndex index, int offset_bytes, half f)              
{
	vector<half, 8> temp;
	read(index, offset_bytes/16*16, temp);
	temp(offset_bytes/sizeof(half)%8)=f;
	write(index, offset_bytes/16*16, temp);
}


template <typename T, int N>
_GENX_ vector<T,N> merge(float f, vector<T,N> v2, vector<ushort,N> v3)              
{
	vector<T,N> r=0;
	r.merge(f,v2,v3);
	return r;
}

)***";                       // NOLINT

boost::future<std::unique_ptr<hal::Library>> cmCompiler::Build(const context::Context& ctx,
                                                               const std::vector<lang::KernelInfo>& kernel_info,
                                                               const hal::proto::HardwareSettings& settings) {
  std::vector<context::proto::ActivityID> kernel_ids;
  std::ostringstream header;

  if (!kernel_info.size()) {
    return boost::make_ready_future(std::unique_ptr<hal::Library>{std::make_unique<cmLibrary>(
        device_state_, std::map<std::string, CmProgram*>{}, std::map<std::string, std::shared_ptr<Emit>>{}, kernel_info,
        std::vector<context::proto::ActivityID>{})});
  }

  context::Activity activity{ctx, "tile::hal::cm::Build"};
  bool cl_khr_fp16 = false;
  bool cl_khr_fp64 = false;

  auto env_cache = env::Get("PLAIDML_CM_CACHE");
  fs::path cache_dir;
  if (env_cache.length()) {
    cache_dir = env_cache;
  }
  std::set<std::string> knames;
  std::map<std::string, CmProgram*> program_map;
  std::map<std::string, std::shared_ptr<Emit>> emit_map;

  for (auto& ki : kernel_info) {
    std::ostringstream code;
    code << header.str();
    context::Activity kbuild{activity.ctx(), "tile::hal::cm::BuildKernel"};

    proto::KernelInfo kinfo;
    kinfo.set_kname(ki.kname);

    if (ki.ktype == lang::KernelType::kZero) {
      kinfo.set_src("// Builtin zero kernel");
    } else if (!knames.count(ki.kfunc->name)) {
      knames.insert(ki.kfunc->name);
      OptimizeKernel(ki, cl_khr_fp16, settings);

      auto pcm = std::make_shared<Emit>(cl_khr_fp16, cl_khr_fp64, ki);

      if (ki.comments.find("= ident") != std::string::npos) {
        pcm->comments_contains_ident = true;
      }

      pcm->Visit(*ki.kfunc);
      std::string src = ki.comments + kernel_header.c_str() + pcm->str();

      auto kname = ki.kname;
      if (is_directory(cache_dir)) {
        kname = kname + "_" + std::to_string(this->knum);
        this->knum++;
      }
      fs::path src_path = (cache_dir / kname).replace_extension("cpp");

      WriteFile(src_path, src);

      fs::path isa_path = (cache_dir / kname).replace_extension("isa");

      CmDevice* pCmDev = device_state_->cmdev();

      std::string cmd = "/home/yangleiz/Desktop/MDF_internal/compiler/bin/cmc ";
      cmd += src_path.string();
      cmd += " -march=GEN9 -isystem ../../compiler/include -o ";
      cmd += isa_path.string();
      auto check_err = system(cmd.c_str());

      if (check_err) {
        check_err = 0;
      }
      CmProgram* program = LoadProgram(pCmDev, isa_path.c_str());

      if (!program) {
        throw std::runtime_error(std::string("Creating an CM program object for ") + ki.kname);
      }

      program_map.emplace(ki.kname, std::move(program));
      emit_map.emplace(ki.kname, pcm);
    } else {
      kinfo.set_src("// Duplicate");
    }

    *(kinfo.mutable_kinfo()) = ki.info;
    kbuild.AddMetadata(kinfo);

    kernel_ids.emplace_back(kbuild.ctx().activity_id());
  }

  cm::cm_Build cm_Build(std::move(activity), device_state_, std::move(program_map), emit_map, kernel_info,
                        std::move(kernel_ids));
  return cm_Build.Start();
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
