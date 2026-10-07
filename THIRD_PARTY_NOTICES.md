# Third-Party Notices

VoxCPMTTS combines components from multiple authors and projects. The root
`LICENSE` applies to the VoxCPMTTS source distribution; it does not replace or
relicense third-party components. Each third-party component remains subject to
its own license terms.

This document is a practical inventory, not legal advice. The license files
and package metadata shipped with each component are authoritative.

## Upstream VoxCPM

This repository is derived from the OpenBMB VoxCPM project and contains
modifications maintained by Hangry Labs.

- Project: [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
- License: Apache License 2.0
- Copyright: OpenBMB and upstream contributors

The Apache License 2.0 text is included in [LICENSE](LICENSE), and upstream
attribution is retained in [NOTICE](NOTICE).

## Model Assets

The standard Docker image downloads and redistributes these model assets for
offline operation:

| Model | Purpose | Declared license |
| --- | --- | --- |
| [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | Speech generation | Apache-2.0 |
| [iic/SenseVoiceSmall](https://www.modelscope.cn/models/iic/SenseVoiceSmall) | Reference-audio transcription | Apache-2.0 |
| [iic/speech_zipenhancer_ans_multiloss_16k_base](https://www.modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base) | Audio denoising | Apache-2.0 |

Model cards and accompanying files are retained in the image's Hugging Face
and ModelScope caches. Model licenses can change between revisions, so release
builds must verify the resolved revisions before publication.

## Python Runtime

The image includes Python and Python packages under their respective licenses.
Major direct runtime dependencies include:

| Component | License family |
| --- | --- |
| CPython | Python Software Foundation License |
| PyTorch and TorchAudio | BSD-style licenses |
| Transformers, Hugging Face Hub, Gradio, ModelScope, Datasets, WeText, Spaces, Safetensors | Apache-2.0 |
| FastAPI, Pydantic, Einops, Inflect, Addict, FunASR | MIT-style licenses |
| Uvicorn, SoundFile, NumPy | BSD and other permissive licenses |
| Librosa | ISC |
| tqdm | MPL-2.0 and MIT |
| simplejson | MIT or AFL-2.1 |
| Matplotlib | Matplotlib license and bundled component licenses |
| Nano-vLLM-VoxCPM and Nano-vLLM | MIT |
| FlashAttention | BSD-3-Clause |

Complete license texts, copyright notices, and bundled-component notices are
retained inside installed Python package metadata, normally under
`/usr/local/lib/python3.13/site-packages/*.dist-info/licenses/`. The CPython
license is retained at `/usr/local/lib/python3.13/LICENSE.txt`.

### Nano-vLLM-VoxCPM and Nano-vLLM

The optional Nano-vLLM Docker targets include software under the MIT License:

```text
MIT License

Copyright (c) 2025 Guoyang Zeng
Copyright (c) 2025 Xingkai Yu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The upstream distribution presents these as separate notices for
Nano-vLLM-VoxCPM (Guoyang Zeng) and Nano-vLLM (Xingkai Yu); both use the same
MIT terms reproduced above.

VoxCPMTTS modifies the pinned Nano-vLLM-VoxCPM runtime so unused CUDA allocator
blocks are released after reference-audio latents have been copied back to CPU.
This prevents different reference-audio lengths from retaining successive GPU
memory high-water allocations.

### FlashAttention

The optional Nano-vLLM Docker targets include FlashAttention under the BSD
3-Clause License:

```text
Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## NVIDIA CUDA Components

CUDA-enabled PyTorch installs NVIDIA runtime libraries including CUDA Runtime,
cuBLAS, cuDNN, cuFFT, cuFile, cuRAND, cuSOLVER, cuSPARSE, cuSPARSELt, NCCL,
NVJitLink, NVSHMEM, NVTX, and CUPTI. Most of these packages are distributed
under NVIDIA proprietary terms; NCCL and NVTX carry separate open-source
licenses.

- Governing terms: [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/)
- Package-specific license files are retained in the corresponding
  `nvidia_*.dist-info/licenses/` directories or package directories.

The Docker images are therefore not distributions whose complete contents are
licensed solely under Apache-2.0. Redistributors must comply with the NVIDIA
terms applicable to the resolved CUDA packages.

## Debian Runtime Components

The Docker images are based on the official Debian-based `python:3.13-slim`
image and install Debian packages, including FFmpeg and libsndfile.

- The Debian FFmpeg build used by the image reports GPL version 2 or later.
  FFmpeg can have different licensing depending on build configuration.
- `libsndfile1` is distributed under LGPL-2.1-or-later, with separately
  licensed bundled components.
- Other operating-system libraries retain their own notices and terms.

Debian copyright files are retained under `/usr/share/doc/*/copyright`, and
common license texts are retained under `/usr/share/common-licenses/`. The
license reported by the installed FFmpeg binary can be viewed with
`ffmpeg -L`.

Corresponding Debian source packages are available from
[Debian Sources](https://sources.debian.org/). Match the source version to the
installed binary version shown by:

```sh
docker run --rm --entrypoint dpkg-query IMAGE \
  -W '-f=${binary:Package}\t${Version}\t${source:Package}\n'
```

In particular, see the Debian source records for
[FFmpeg](https://sources.debian.org/src/ffmpeg/) and
[libsndfile](https://sources.debian.org/src/libsndfile/). Anyone redistributing
an image must continue to provide the notices and corresponding source access
required by the applicable GPL and LGPL versions.

## Inspecting a Built Image

Licenses are resolved when the image is built. To inventory an exact image:

```sh
docker run --rm --entrypoint dpkg-query IMAGE \
  -W '-f=${binary:Package}\t${Version}\t${source:Package}\n'

docker run --rm --entrypoint python IMAGE -c \
  'from importlib.metadata import distributions; print("\n".join(sorted("{}=={}".format(d.metadata.get("Name"), d.version) for d in distributions())))'
```

Before publishing a new image, review package metadata, retained license files,
model cards, and model revisions again. Dependency upgrades can change the
applicable terms.
