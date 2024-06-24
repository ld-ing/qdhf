# Quality Diversity through Human Feedback

### [Project Page](https://liding.info/qdhf/) | [Paper](https://arxiv.org/abs/2310.12103) | [Demo](https://huggingface.co/spaces/jennyzzt/quality-diversity-through-human-feedback) (new) | [Talk](https://neurips.cc/virtual/2023/83789) | [Tutorial](https://docs.pyribs.org/en/stable/tutorials/qdhf.html) | [Cite](#citation)

#### Official Python implementation of [Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization](https://arxiv.org/abs/2310.12103) ([ICML 2024](https://icml.cc/virtual/2024/poster/34789) & Spotlight at [NeurIPS 2023 ALOE Workshop](https://sites.google.com/view/aloe2023/accepted-papers))

[Li Ding](https://liding.info/), [Jenny Zhang](https://www.jennyzhangzt.com/), [Jeff Clune](http://jeffclune.com/), [Lee Spector](https://lspector.github.io/), [Joel Lehman](http://joellehman.com/)

**TL;DR:** QDHF enhances QD algorithms by inferring diversity metrics from human judgments of similarity, surpassing state-of-the-art methods in automatic diversity discovery in robotics & RL tasks and significantly improving performance in open-ended generative tasks.

![teaser](teaser.jpg)
<p align="center">
QDHF (right) improves the diversity in text-to-image generation results compared to best-of-N (left) using Stable Diffusion. 
</p>

## Updates
- **2024-06-24**: Release of the [QDHF Gradio Demo](https://huggingface.co/spaces/jennyzzt/quality-diversity-through-human-feedback) on [Hugging Face](https://huggingface.co/).
- **2024-03-14**: Release of the [QDHF tutorial](https://docs.pyribs.org/en/stable/tutorials/qdhf.html) in [pyribs](https://pyribs.org/).
- **2023-12-13**: Initial release of the codebase.

## Demo (new)

We have released a [Gradio Demo](https://huggingface.co/spaces/jennyzzt/quality-diversity-through-human-feedback) on [Hugging Face](https://huggingface.co/). This user-friendly interface enables effortless exploration of QDHF without any coding requirements. Special thanks to [Jenny Zhang](https://www.jennyzhangzt.com/) for her contributions!

## Tutorial

We have released a tutorial: [Incorporating Human Feedback into Quality Diversity for Diversified Text-to-Image Generation](https://docs.pyribs.org/en/stable/tutorials/qdhf.html), together with the [pyribs](https://pyribs.org/) team. This tutorial features a lightweight version of QDHF and runs on Google Colab in ~1 hour. Dive into the tutorial to explore how QDHF enhances GenAI models with diversified, high-quality responses and apply these insights to your projects!

## Requirements

To install the requirements, run:
```
pip install -r requirements.txt
```

## Usage

For each experiment, we provide a `main.py` script to run the experiment. For example, to run the robotic arm experiment, run:
```
cd arm
python3 main.py
```
Replace `arm` with the name of the experiment you want to run.

<a name="citation"></a>
## Citation

If you find our work or any of our materials useful, please cite our paper:
```
@inproceedings{
      ding2024quality,
      title={Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization},
      author={Li Ding and Jenny Zhang and Jeff Clune and Lee Spector and Joel Lehman},
      booktitle={Forty-first International Conference on Machine Learning},
      year={2024},
      url={https://openreview.net/forum?id=9zlZuAAb08}
}
```

## License
This project is under the [MIT License](LICENSE).

##  Acknowledgments
The main structure of this code is modified from the [DQD](https://github.com/icaros-usc/dqd/tree/main). Each experiment contains its own modified version of [pyribs](https://pyribs.org/), a quality diversity optimization library. The maze navigation experiment uses a modified version of [Kheperax](https://github.com/adaptive-intelligent-robotics/Kheperax). The LSI experiment uses Stable Diffusion ([huggingface/diffusers](https://github.com/huggingface/diffusers)), [OpenAI CLIP](https://github.com/openai/CLIP), and [DreamSim](https://github.com/ssundaram21/dreamsim). The funding acknowledgments are disclosed in the paper.
