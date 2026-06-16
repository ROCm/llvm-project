# llvm-project-docs

> [!NOTE]
> The published documentation is available at [llvm-project documentation](https://rocm.docs.amd.com/projects/llvm-project/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `./docs` folder of this GitHub repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

Documentation repository for [llvm-project](https://github.com/ROCm/llvm-project)

This repository consolidates the documentation for llvm.  Compiler topics originally in [https://github.com/ROCm/ROCm](https://github.com/ROCm/ROCm) have been moved to this repo.

In addition, the pre-built HTML pages from rocm-llvm-docs packages will be extracted and hosted here so users can access docs specific to the llvm release associated with each ROCm release.

## How to build documentation locally

Run the following steps to build the base documentation site:

```bash
cd docs
pip3 install -r sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```
