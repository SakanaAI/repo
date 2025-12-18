<br />
<div align="center">
  <!-- <a href="https://github.com/your_username/repo_name">
    <img src="https://via.placeholder.com/100" alt="Logo" width="80" height="80">
  </a> -->

  <h3 align="center">RePo: Language Models with Context Re-Positioning</h3>

  <p align="center">
    An light-weight module that allows LLMs to re-structure the context adaptively.
    <!-- <br /> -->
    <!-- <a href="https://arxiv.org/abs/xxxx.xxxxx"><strong>Explore the docs »</strong></a> -->
    <!-- <br /> -->
    <br />
    <a href="https://arxiv.org/abs/2512.14391"><img src="https://img.shields.io/badge/arXiv-2512.14391-b31b1b.svg?style=flat-square" alt="arXiv"></a>
    <a href="https://huggingface.co/SakanaAI/RePo-OLMo2-1B-stage2-L5"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue?style=flat-square" alt="Hugging Face"></a>
    <!-- <a href="https://colab.research.google.com/"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> -->
    <!-- <a href="LICENSE"><img src="https://img.shields.io/github/license/SakanaAI/repo?style=flat-square" alt="License"></a> -->
  </p>
</div>

---

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#-abstract">Abstract</a></li>
    <li><a href="#-installation">Installation</a></li>
    <li><a href="#-usage">Usage</a></li>
    <li><a href="#-training">Training</a></li>
    <li><a href="#-citation">Citation</a></li>
    <li><a href="#-acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## 🔥 News
* **[2025.12]** We add an interactive demo to visualize the assigned positions. Please find it in `./visual`!
* **[2025.12]** We have released the [training code](#-training) and [evaluation scripts](#-usage)!
* **[2025.12]** Pre-trained models (based on OLMo-2 1B) are now available on [Hugging Face](https://huggingface.co/SakanaAI/RePo-OLMo2-1B-stage2-L5).
* **[2025.12]** The paper "RePo: Language Models with Context Re-Positioning" is released on [arXiv](https://arxiv.org/abs/2512.14391).

## 🧩 Abstract

<!-- ![Method Overview](https://via.placeholder.com/800x300?text=Model+Architecture+or+Demo+GIF) -->

In-context learning is fundamental to modern Large Language Models (LLMs); however, prevailing architectures impose a rigid and fixed contextual structure by assigning linear or constant positional indices. Drawing on Cognitive Load Theory (CLT), we argue that this uninformative structure increases extraneous cognitive load, consuming finite working memory capacity that should be allocated to deep reasoning and attention allocation. To address this, we propose RePo, a novel mechanism that reduces extraneous load via context re-positioning. Unlike standard approaches, RePo utilizes a differentiable module, $f_\phi$, to assign token positions that capture contextual dependencies, rather than replying on pre-defined integer range. By continually pre-training on the OLMo-2 1B backbone, we demonstrate that RePo significantly enhances performance on tasks involving noisy contexts, structured data, and longer context length, while maintaining competitive performance on general short-context tasks. Detailed analysis reveals that RePo successfully allocate higher attention to distant but relevant information, assign positions in dense and non-linear space, and capture the intrinsic structure of the input context. 

> This is the initial repository for the research project **RePo**. Please feel free to open issues if you have any questions or find any mistakes.

<!-- ## 🦁 Model Zoo

We provide pre-trained weights on Hugging Face.

| Model | Params | Context Length | MMLU Score | Hugging Face |
| :--- | :---: | :---: | :---: | :--- |
| **Model-Small** | 7B | 4k | 45.2 | [Link](https://huggingface.co/) |
| **Model-Base** | 13B | 8k | 56.8 | [Link](https://huggingface.co/) |
| **Model-Chat** | 13B | 8k | 58.1 | [Link](https://huggingface.co/) | -->

## 🛠️ Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/SakanaAI/repo
    cd repo
    ````

2.  **Setup for Evaluation**

    ```bash
    # We tested this setup on H100 and 6000Ada
    # in ./repo
    conda create -n olmes python=3.11

    ### enable only if you have CUDA > 12.4
    # conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit

    ### install torch
    please adjust --index-url according to your CUDA
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

    ### install vLLM with RePo
    cd vllm
    python use_existing_torch.py
    pip install -r requirements/build.txt
    mkdir -p vllm/vllm_flash_attn
    pip install -e . --no-build-isolation

    ### install transformers with RePo
    cd ../transformers
    pip install -e '.[torch]' --no-build-isolation

    ### install test suites
    cd ../olmes
    pip install -e . --no-build-isolation
    ```

3.  **Setup for Train**

    ```bash
    # We tested this setup on H100
    # in ./repo
    cd OLMo

    ### install OLMo with RePo
    conda env create -f environment.yml
    conda activate olmo
    pip install flash-attn==2.7.4.post1
    pip install -e .[all]
    ```

## 💻 Usage

### Quick Inference

Please download the checkpoints from [huggingface](https://huggingface.co/huayangli/OLMo2-1B-RePo) in adavance:

```bash
cd olmes
bash eval_ruler.sh
```

## 🏋️ Training

Please take a look at the script `OLMo/batch_run_stage2_1b.sh`, you need to replace the placeholder to the state-2 data by your real data path, following the instruction of [OLMo](https://github.com/allenai/OLMo).

```bash
cd OLMo
SLURM_ARRAY_TASK_ID=2 bash batch_run_stage2_1b.sh -d $YOUR_DATA_DIR
```


## 📜 Citation

If you find this project useful, please cite our paper:

```bibtex
@article{sakana2025repo,
  title={RePo: Language Models with Context Re-Positioning},
  author={Huayang Li, Tianyu Zhao, and Richard Sproat},
  year={2025},
  eprint={2512.14391},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2512.14391},
}
```

## 🙏 Acknowledgments

  * We utilized code from [OLMo](https://github.com/allenai/OLMo), [olmes](https://github.com/allenai/olmes), [vLLM](https://github.com/vllm-project/vllm), and [transformers](https://github.com/huggingface/transformers).

<!-- end list -->