#!/usr/bin/env python3
"""Compile the patched CPU budget/cache contract without CUDA on the host.

Uses the actual upstream smart-resize and hash implementations and extracts the
patched budget/identity methods unchanged. GPU engine integration is a separate
on-device check; these tests do not claim numerical image equivalence.
"""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external/TensorRT-Edge-LLM"
PATCH = ROOT / "patches/cosmos-runtime-image-token-budget.patch"


def function(source, signature):
    start = source.index(signature)
    body = source.index("{", start)
    depth = 1
    end = body + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


class ImageBudgetPatchTest(unittest.TestCase):
    def test_native_host_contract(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler or not UPSTREAM.is_dir():
            self.skipTest("Requires C++ compiler and the pinned public backend checkout")
        with tempfile.TemporaryDirectory(prefix="cosmos-image-budget-") as temp:
            tree = Path(temp)
            paths = re.findall(r"^diff --git a/(\S+) b/\S+$", PATCH.read_text(), re.M)
            for path in paths:
                dest = tree / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(UPSTREAM / path, dest)
            # The working checkout may already have this exact patch applied.
            has_patch = 'maxImageTokensPerImage{0}' in (tree / 'cpp/runtime/imageUtils.h').read_text()
            if not has_patch:
                subprocess.run(["patch", "-s", "-p1", "-d", str(tree)], input=PATCH.read_bytes(), check=True)
            stub = tree / "stubs/common"
            stub.mkdir(parents=True)
            (stub / "checkMacros.h").write_text('#include <stdexcept>\n#define ELLM_CHECK(c,m) do { if (!(c)) throw std::runtime_error(m); } while(false)\n')
            (stub / "logger.h").write_text('#define LOG_WARNING_IF(...) do {} while(false)\n')
            qwen = (tree / 'cpp/multimodal/qwen2/qwenViTRunner.cpp').read_text()
            cosmos = (tree / 'cpp/multimodal/qwen3/qwen3vlViTRunner.cpp').read_text()
            rank = (tree / 'cpp/runtime/llmRankRuntime.cpp').read_text()
            identity = rank[rank.index('                        std::string const identity ='):rank.index('                        imageHashes.push_back(hashOpaqueIdentity(identity));')]
            identity = identity.strip() + '\nreturn hashOpaqueIdentity(identity);\n'
            contract = r'''
#include <cassert>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include "common/checkMacros.h"
#include "multimodal/common/imageUtils.h"
#include "runtime/state/contextCache/blockHash.h"
#include "profiling/metrics.h"
namespace trt_edgellm {
bool enabled = false;
bool getProfilingEnabled() noexcept { return enabled; }
void setProfilingEnabled(bool value) noexcept { enabled = value; }
namespace rt {
namespace imageUtils {
struct ImageData { int64_t maxImageTokensPerImage=0,width=1280,height=720,frames=1,channels=3; bool doResize=true,isVideo=false; };
}
struct QwenViTRunner {
 struct {int64_t minImageTokensPerImage=4,maxImageTokensPerImage=512,patchSize=16,mergeSize=2,temporalPatchSize=1;} mConfig;
 virtual std::tuple<int64_t,int64_t> getResizedImageSize(int64_t,bool,int64_t,int64_t,int64_t=200,int64_t=0);
 std::tuple<int64_t,int64_t> getImageResizeTarget(imageUtils::ImageData const&);
};
struct Qwen3VLViTRunner : QwenViTRunner {
 std::tuple<int64_t,int64_t> getResizedImageSize(int64_t,bool,int64_t,int64_t,int64_t=200,int64_t=0) override;
};
'''
            contract += function(qwen, 'std::tuple<int64_t, int64_t> QwenViTRunner::getResizedImageSize(') + '\n'
            contract += function(qwen, 'std::tuple<int64_t, int64_t> QwenViTRunner::getImageResizeTarget(') + '\n'
            contract += function(cosmos, 'std::tuple<int64_t, int64_t> Qwen3VLViTRunner::getResizedImageSize(') + '\n'
            contract += 'Hash128 identity(imageUtils::ImageData const& img) { auto const pixels=hashOpaqueIdentity("same decoded pixels");\n' + identity + '}\n}}\n'
            contract += r'''
using namespace trt_edgellm;
using namespace trt_edgellm::rt;
int main() {
 Qwen3VLViTRunner runner;
 imageUtils::ImageData image;
 auto original=runner.getImageResizeTarget(image);
 image.maxImageTokensPerImage=512;
 assert(original==runner.getImageResizeTarget(image));
 int cases=0;
 for (int64_t cap : {4,16,64,128,256,320,511,512}) {
  image.maxImageTokensPerImage=cap;
  for (auto size : std::vector<std::pair<int64_t,int64_t>>{{1280,720},{720,1280},{640,640},{64,64},{2000,10},{10,2000},{1920,1080}}) {
   image.width=size.first;image.height=size.second;
   auto [h,w]=runner.getImageResizeTarget(image);
   assert(h%32==0 && w%32==0 && h*w/1024>=4 && h*w/1024<=cap);
   ++cases;
  }
 }
 auto rejected=[&](){try {runner.getImageResizeTarget(image);return false;}catch(std::runtime_error const&){return true;}};
 image.width=1280;image.height=720;
 for(int64_t cap : {-1,1,3,513,4096}) {image.maxImageTokensPerImage=cap;assert(rejected());}
 image.maxImageTokensPerImage=320;image.isVideo=true;assert(rejected());image.isVideo=false;
 image.frames=2;assert(rejected());image.frames=1;
 image.doResize=false;assert(rejected());
 image.width=640;image.height=512;assert(std::get<0>(runner.getImageResizeTarget(image))==512);
 image.width=672;assert(rejected());image.doResize=true;
 auto key320=identity(image);image.maxImageTokensPerImage=512;assert(key320!=identity(image));
 auto key512=identity(image);image.doResize=false;assert(key512!=identity(image));
 image.doResize=true;image.width=512;image.height=672;assert(key512!=identity(image));
 metrics::MultimodalMetrics metrics;
 assert(!getProfilingEnabled());metrics.recordRun(1,299);
 assert(metrics.observedImageRuns==1 && metrics.observedImageTokens==299 && metrics.getTotalRuns()==0);
 metrics.recordRun(0,0,1,8);assert(metrics.observedImageRuns==1);
 setProfilingEnabled(true);metrics.recordRun(1,504);
 assert(metrics.observedImageRuns==2 && metrics.observedImageTokens==803 && metrics.getTotalRuns()==1);
 std::cout << cases << " native smart-resize cases; invalid limits, video/raw boundaries, cache identity, counters passed\n";
}
'''
            cpp = tree / "contract.cpp"
            cpp.write_text(contract)
            binary = tree / "contract"
            subprocess.run([compiler, "-std=c++17", "-O0", "-I"+str(tree/"stubs"), "-I"+str(tree/"cpp"), "-I"+str(UPSTREAM/"cpp"), str(cpp), str(UPSTREAM/"cpp/multimodal/common/imageUtils.cpp"), str(UPSTREAM/"cpp/runtime/state/contextCache/blockHash.cpp"), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
