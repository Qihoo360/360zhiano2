---
license: apache-2.0
language:
- zh
- en
library_name: transformers
tags:
- qihoo360
- 奇虎360
- zhinao
- 360Zhinao
- pretrain
---

<p align="left">
    <a href="./README_CN.md">中文</a> ｜ &nbsp English</a>&nbsp
</p>
<br>

<div align="center">
<h1>
  360Zhinao2 (360智脑)
</h1>
</div>
<div align="center">
    🤗 <a href="https://huggingface.co/qihoo360">HuggingFace</a>&nbsp&nbsp | &nbsp&nbsp
    🤖 <a href="https://modelscope.cn/organization/360zhinao">ModelScope</a>&nbsp&nbsp ｜ &nbsp&nbsp
    💬 <a href="./assets/WeChat.png">WeChat (微信)</a>&nbsp&nbsp 
</div>
<br>
<p align="center">
 Feel free to visit 360Zhinao's official website<a href="https://ai.360.com"> https://ai.360.com</a> for more experience.
</p>

<br>

# Introduction
 🎉🎉🎉 We released the 360Zhinao2 model series:
 - **360Zhinao2-7B-Base**
 - **360Zhinao2-7B-Chat-4K**
 - **360Zhinao2-7B-Chat-32K**
 - **360Zhinao2-7B-Chat-360K**

Notable features of our 360Zhinao models are:

- **Base Model:** Using popular two-stage training method, In the first stage we totally train 10T tokens with a cosine learning rate schedule. In the second stage we increase the proportion of high-quality data and totally train 100B tokens, with the learning rate decaying directly to 0. The total training data for 360Zhinao2-7B amounts to 10.1T tokens.
- **Chat Models:** Powerful chat capabilities and three context lengths of 4K, 32K and 360K. 

<br>

# News and Updates
- [2024.11.18] 🔥🔥🔥We release 360Zhinao2-7B, providing access to both the Base model and Chat models with text lengths of 4K, 32K, and 360K.
- [2024.05.23] We released two models, 360Zhinao-search and 360Zhinao-1.8B-Reranking, which ranked first respectively in the Retrieval and Reranking tasks of [C-MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) .
- [2024.05.20] We extended llama3 and released **llama3-8B-360Zhinao-360k-Instruct**<a href="https://huggingface.co/qihoo360/llama3-8B-360Zhinao-360k-Instruct">🤗</a>
- [2024.04.12] We released **360Zhinao-7B** v1.0, including the base model and three chat models with context lengths 4K, 32K and 360K. 
Technical report is on [arXiv](https://arxiv.org/abs/2405.13386).

<br>

# Table of contents
- [Download URL](#Download-URL)
- [Model Evaluation](#Model-Evaluation)
- [Quickstart](#Quickstart)
- [Model Inference](#Model-Inference)
- [Model Finetune](#Model-Finetune)
- [License](#License)

<br>

# Download URL

| Size | Model | BF16 | Int4|
|-|-|-|-|
| 7B | 360Zhinao2-7B-Base | <a href="https://modelscope.cn/models/360zhinao/360Zhinao2-7B-Base/summary">🤖</a>  <a href="https://huggingface.co/qihoo360/360Zhinao2-7B-Base">🤗</a> |  |
| 7B | 360Zhinao2-7B-Chat-4K | <a href="https://modelscope.cn/models/360zhinao/360Zhinao2-7B-Chat-4K/summary">🤖</a>  <a href="https://huggingface.co/qihoo360/360Zhinao2-7B-Chat-4K">🤗</a> | <a href="https://modelscope.cn/models/360zhinao/360Zhinao2-7B-Chat-4K-Int4/summary">🤖</a>  <a href="https://huggingface.co/qihoo360/360Zhinao2-7B-Chat-4K-Int4">🤗</a> |
| 7B | 360Zhinao2-7B-Chat-32K | <a href="https://modelscope.cn/models/360zhinao/360Zhinao2-7B-Chat-32K/summary">🤖</a>  <a href="https://huggingface.co/qihoo360/360Zhinao2-7B-Chat-32K">🤗</a> | <a href="https://modelscope.cn/models/360zhinao/360Zhinao2-7B-Chat-32K-Int4/summary">🤖</a>  <a href="https://huggingface.co/qihoo360/360Zhinao2-7B-Chat-32K-Int4">🤗</a> |
| 7B | 360Zhinao2-7B-Chat-360K | <a href="https://modelscope.cn/models/360zhinao/360Zhinao2-7B-Chat-360K/summary">🤖</a>  <a href="https://huggingface.co/qihoo360/360Zhinao2-7B-Chat-360K">🤗</a> | <a href="https://modelscope.cn/models/360zhinao/360Zhinao2-7B-Chat-360K-Int4/summary">🤖</a>  <a href="https://huggingface.co/qihoo360/360Zhinao2-7B-Chat-360K-Int4">🤗</a> |

<br>

# Model Evaluation
## Base Model
We used the open-source tool OpenCompass to evaluate the model and compared it with open-source models under 10B from the past six months. The 360Zhinao2-7B model is competive. The 360Zhinao2-7B model performs well on Chinese benchmarks such as CEval, C3 and LCSTS. The average socres of Chinese benchmarks is No 1. It also ranks No 1 on Math which is a challenging competition math dataset. **The 360Zhinao2-7B model has advantages in Chinese benchmark and challenging competition math.**

<table>
	<tr>
	    <td>Type</td><td>Datasets</td><td>language</td><td>glm4-9b</td><td>Qwen2.5-7B</td><td>internlm2.5-7b</td><td>Yi1.5-9B</td><td>gemma2-9b</td><td>Llama3.1-8B</td><td>360Zhinao2-7B</td>
	</tr>
	<tr>
	    <td rowspan="5">Exam</td><td>ceval</td><td>zh</td><td>75.83</td><td>81.41</td><td>77.71</td><td>73.51</td><td>56.36</td><td>51.67</td><td><strong>83.04</strong></td>
	</tr>
    <tr>
        <td>mmlu</td><td>en</td><td>75.5</td><td>75.5</td><td>71.55</td><td>71.43</td><td>72.22</td><td>66.75</td><td>67.84</td>
    </tr>
    <tr>
        <td>cmmlu</td><td>zh</td><td>74.24</td><td>81.79</td><td>78.77</td><td>74.2</td><td>58.89</td><td>52.49</td><td>73.8</td>
    </tr>
    <tr>
        <td>ARC-c</td><td>en</td><td>94.92</td><td>80</td><td>85.08</td><td>87.46</td><td>77.63</td><td>80.68</td><td>87.12</td>
    </tr>
    <tr>
        <td>ARC-e</td><td>en</td><td>98.41</td><td>84.83</td><td>95.24</td><td>94.53</td><td>78.84</td><td>89.77</td><td>92.77</td>
    </tr>
    <tr>
        <td rowspan="2">Language</td><td>WiC</td><td>en</td><td>51.57</td><td>52.82</td><td>50.78</td><td>50.63</td><td>50.47</td><td>50</td><td>49.84</td>
    </tr>
    <tr>
        <td>WSC</td><td>en</td><td>68.27</td><td>68.27</td><td>69.23</td><td>66.35</td><td>68.27</td><td>67.31</td><td>65.38</td>
    </tr>
    <tr>
        <td rowspan="2">Knowledge</td>
        <td>BoolQ</td><td>en</td><td>81.8</td><td>83.88</td><td>89.51</td><td>84.46</td><td>85.6</td><td>82.2</td><td>88.29</td>
    </tr>
    <tr>
        <td>commonsense_qa</td><td>en</td><td>71.17</td><td>73.22</td><td>68.55</td><td>71.58</td><td>68.47</td><td>71.25</td><td>69.78</td>
    </tr>
    <tr>
        <td rowspan="6">Understanding</td>
        <td>C3</td><td>zh</td><td>91.51</td><td>92</td><td>93.04</td><td>85.86</td><td>81.64</td><td>83.51</td><td><strong>93.26</strong></td>
    </tr>
    <tr>
        <td>race-middle</td><td>en</td><td>91.99</td><td>91.02</td><td>92.06</td><td>91.16</td><td>88.09</td><td>81.69</td><td>90.46</td>
    </tr>
    <tr>
        <td>race-high</td><td>en</td><td>90.71</td><td>87.91</td><td>90.08</td><td>88.34</td><td>82.08</td><td>78.73</td><td>86.74</td>
    </tr>
    <tr>
        <td>lcsts</td><td>zh</td><td>18.29</td><td>15.82</td><td>15.96</td><td>16.49</td><td>10.62</td><td>17.29</td><td><strong>18.61</strong></td>
    </tr>
    <tr>
        <td>eprstmt-dev</td><td>zh</td><td>91.88</td><td>86.88</td><td>91.25</td><td>91.88</td><td>48.12</td><td>83.12</td><td>90</td>
    </tr>
    <tr>
        <td>lambada</td><td>en</td><td>71.67</td><td>71.14</td><td>69.98</td><td>70.64</td><td>75.43</td><td>74.23</td><td>72.56</td>
    </tr>
    <tr>
        <td rowspan="3">Reasoning</td>
        <td>hellaswag</td><td>en</td><td>70.25</td><td>72.76</td><td>70.38</td><td>71.55</td><td>66.83</td><td>74.65</td><td>71.49</td>
    </tr>
    <tr>
        <td>siqa</td><td>en</td><td>81.73</td><td>72.52</td><td>78.97</td><td>76.2</td><td>58.96</td><td>64.18</td><td>77.12</td>
    </tr>
    <tr>
        <td>bbh</td><td>en</td><td>73.68</td><td>54.63</td><td>59.43</td><td>67.86</td><td>68.45</td><td>59.9</td><td>46.54</td>
    </tr>
    <tr>
        <td rowspan="2">Code</td>
        <td>humaneval</td><td>en</td><td>69.51</td><td>75</td><td>60.37</td><td>26.22</td><td>5.49</td><td>27.44</td><td>60.98</td>
    </tr>
    <tr>
        <td>mbpp</td><td>en</td><td>60</td><td>60</td><td>43.6</td><td>56.8</td><td>51.2</td><td>42.6</td><td>54</td>
    </tr>
    <tr>
        <td rowspan="2">Math</td>
        <td>math</td><td>en</td><td>26.86</td><td>38</td><td>27.14</td><td>27.06</td><td>28.52</td><td>15.32</td><td><strong>38.34</strong></td>
    </tr>
    <tr>
        <td>gsm8k</td><td>en</td><td>78.54</td><td>79.76</td><td>52.54</td><td>71.11</td><td>73.09</td><td>56.25</td><td>75.51</td>
    </tr>
    <tr>
        <td rowspan="2">Overall</td>
        <td>avg_zh</td><td></td><td>70.35</td><td>71.58</td><td>71.35</td><td>68.39</td><td>51.13</td><td>57.62</td><td><strong>71.74</strong></td>
    </tr>
    <tr>
        <td>avg_all</td><td></td><td>73.11</td><td>71.78</td><td>69.60</td><td>68.88</td><td>61.60</td><td>62.32</td><td>70.61</td>
    </tr>
</table>


<br>


## Chat Model

### Post-Training Data
360's proprietary general fine-tuning dataset consists of 500,000 samples. This dataset considers various skills and 360's vertical business data, with the following generation methods:

1. **Data Diversity**: Layered sampling based on 360's proprietary tagging system, considering domain, intent, difficulty, and length to ensure instruction diversity.
2. **Data Quality**: Using open-source data and proprietary preference-ordered data to train 360gpt-pro-rm (reward benchmark score of 92.59). This model is used for sample screening to filter out low-quality responses.
3. **Complex Instruction Evolution**: Optimizing complex instructions through evolutionary methods to enhance instruction-following capabilities.

### Training Methods
1. **Full Parameter Fine-Tuning**
   
   Based on the general post-training data, full parameter fine-tuning is performed, and the optimal checkpoint is selected as sft-base.

2. **LoRA Off-policy DPO**
   
   Using human-labeled preference pairs, LoRA fine-tuning is applied to the sft-base model, followed by LoRA DPO training.

3. **Iterative On-Policy DPO**
   
   The sft-base model samples multiple answers on training prompts, and 360gpt-pro-rm scores them. The highest and lowest scoring answers form each pair, and such pairs are used for DPO training. This on-policy DPO method is iteratively used to improve model performance.

4. **Model Merging**
   
   Automatic evaluations on 360's white-box evaluation set v4 revealed that different models excel in different skills. A model merging scheme was considered to result in the final chat model.

### Model Performance
We evaluated the 360Zhinao2-7B-Chat-4k model on IFEval, MT-bench, CF-bench which are all popular benchmark to evaluate chat model ability. The evaluation results show that 360Zhinao2-7B is highly competitive. Especially the IFEval (prompt strict) score get the highest score among open-source 7B models. only a little worse than GLM4-9B. The detailed results are shown in the table below:

| Model                | MT-bench | IFEval(strict prompt) | CFBench(CSR,ISR,PSR) |      |      |
|----------------------|----------|-----------------------|----------------------|------|------|
| Qwen2.5-7B-Instruct  | **8.07** | 0.556                 | **0.81**             | 0.46 | 0.57 |
| Yi-9B-16k-Chat       | 7.44     | 0.455                 | 0.75                 | 0.4  | 0.52 |
| GLM4-9B-Chat         | **8.08** | **0.634**             | **0.82**             | 0.48 | 0.61 |
| InternLM2.5-7B-Chat  | 7.39     | 0.540                 | 0.78                 | 0.4  | 0.54 |
| 360Zhinao2-7B-Chat-4k| 7.86     | **0.577**             | 0.8                  | 0.44 | 0.57 |

### Long Context Fine-Tuning
Similar to the method used during the open-sourcing of 360Zhinao1, we expanded the RoPE base to 1,000,000 and 50,000,000, sequentially concatenated SFT data of mixed long and short texts to 32k and 360k. By combining techniques like gradient checkpointing, ZeRO3 offload, and ring attention, we fine-tuned models to achieve 32k and 360k long context capabilities. These models ranked in the top tier across various 32k benchmarks.

| Model                        | LooGLE-Long Dependency QA | Loong-Set 1 (32k) | LongBench-Chat (32k cutoff) | LEval-96 question subset | LEval-closed ended |
|------------------------------|-----------------|-------------------|--------------------------|--------------------|------------------|
| GLM4-9B-Chat                 | 0.36            | 55.24             | 6.60                     | 0.49               | 63.96            |
| InternLM2.5-7B-Chat          | 0.39            | 42.76             | 5.70                     | 0.44               | 61.64            |
| 360Zhinao2-7B-Chat-32k       | 0.33            | 39.37             | 5.44                     | 0.44               | 60.48            |
| 360Zhinao2-7B-Chat-360k      | 0.34            | 32.16             | 5.08                     | 0.38               | 53.00            |
| Yi-1.5-9B-Chat               | 0.25            | 32.77             | 4.70                     | 0.37               | 56.22            |

<br>




# Quickstart
We provide simple examples illustrating the use of 360Zhinao2-7B-Base and 360Zhinao2-7B-Chat on 🤖ModelScope and 🤗Transformers.

## Dependency Installation
- python >= 3.8
- pytorch >= 2.0
- transformers >= 4.37.2
- CUDA >= 11.4

```shell
pip install -r requirements.txt 
```

Optionally, we recommend installing Flash-Attention 2 to improve performance and reduce memory footprint.

>flash-attn >= 2.3.6
```shell
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn==2.3.6
```

## 🤗 Transformers
### Demonstration of Base Model Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

MODEL_NAME_OR_PATH = "qihoo360/360Zhinao2-7B-Base"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME_OR_PATH, 
    trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_OR_PATH,
    device_map="auto",
    trust_remote_code=True)

generation_config = GenerationConfig.from_pretrained(
    MODEL_NAME_OR_PATH,
    trust_remote_code=True)

inputs = tokenizer('中国二十四节气\n1. 立春\n2. 雨水\n3. 惊蛰\n4. 春分\n5. 清明\n', return_tensors='pt')
inputs = inputs.to(model.device)

pred = model.generate(input_ids=inputs["input_ids"], generation_config=generation_config)
print("outputs:\n", tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
```
### Demonstration of Chat Model Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

MODEL_NAME_OR_PATH = "qihoo360/360Zhinao2-7B-Chat-4K"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME_OR_PATH, 
    trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_OR_PATH,
    device_map="auto",
    trust_remote_code=True)

generation_config = GenerationConfig.from_pretrained(
    MODEL_NAME_OR_PATH,
    trust_remote_code=True)

messages = []
#round-1
messages.append({"role": "user", "content": "介绍一下刘德华"})
response = model.chat(tokenizer=tokenizer, messages=messages, generation_config=generation_config)
messages.append({"role": "assistant", "content": response})
print(messages)

#round-2
messages.append({"role": "user", "content": "他有什么代表作？"})
response = model.chat(tokenizer=tokenizer, messages=messages, generation_config=generation_config)
messages.append({"role": "assistant", "content": response})
print(messages)
```

## 🤖 ModelScope
### Demonstration of Base Model Inference

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig

MODEL_NAME_OR_PATH = "qihoo360/360Zhinao2-7B-Base"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME_OR_PATH, 
    trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_OR_PATH,
    device_map="auto",
    trust_remote_code=True)

generation_config = GenerationConfig.from_pretrained(
    MODEL_NAME_OR_PATH,
    trust_remote_code=True)

inputs = tokenizer('中国二十四节气\n1. 立春\n2. 雨水\n3. 惊蛰\n4. 春分\n5. 清明\n', return_tensors='pt')
inputs = inputs.to(model.device)

pred = model.generate(input_ids=inputs["input_ids"], generation_config=generation_config)
print("outputs:\n", tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
```

### Demonstration of Chat Model Inference

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig

MODEL_NAME_OR_PATH = "qihoo360/360Zhinao2-7B-Chat-4K"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME_OR_PATH, 
    trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_OR_PATH,
    device_map="auto",
    trust_remote_code=True)

generation_config = GenerationConfig.from_pretrained(
    MODEL_NAME_OR_PATH,
    trust_remote_code=True)

messages = []
#round-1
messages.append({"role": "user", "content": "介绍一下刘德华"})
response = model.chat(tokenizer=tokenizer, messages=messages, generation_config=generation_config)
messages.append({"role": "assistant", "content": response})
print(messages)

#round-2
messages.append({"role": "user", "content": "他有什么代表作？"})
response = model.chat(tokenizer=tokenizer, messages=messages, generation_config=generation_config)
messages.append({"role": "assistant", "content": response})
print(messages)
```

## CLI Demo
Use terminal for command-line interface:

```shell
python cli_demo.py
```
<p align="center">
    <img src="assets/cli_demo.gif" width="600" />
<p>

Note: for Mac users, `device = 'mps'` is not supported yet.

## Web Demo

```shell
streamlit run web_demo.py
```
<p align="center">
    <img src="assets/web_demo.gif" width="600" />
<p>

## API Demo
Launch api:
```shell
python openai_api.py
```

Then request with parameters:
```shell
curl 'http://localhost:8360/v1/chat/completions' \
-H 'Content-Type: application/json' \
-d '{
    "max_new_tokens": 200,
    "do_sample": true,
    "top_k": 0,
    "top_p": 0.8,
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好"}
    ]
}'
```

<br>

# Model Inference
## Quantization
We provide quantization schemes based on AutoGPTQ and release the Int4 quantization models. 

## Deployment
### vLLM Installation
We recommend using `vLLM==0.3.3`.

If you are using **CUDA 12.1 and PyTorch 2.1**, you can install vLLM directly with:
```shell
pip install vllm==0.3.3
```

Otherwise, please refer to the official vLLM [Installation Instructions](https://docs.vllm.ai/en/latest/getting_started/installation.html).

After installation, perform the following steps:
1. Copy `vllm/zhinao.py` into `vllm/model_executor/models` in your vllm installation directory (in python/conda env).
2. Copy `vllm/serving_chat.py` into `vllm/entrypoints/openai` in your vllm installation directory.
3. Then add a line in `vllm/model_executor/models/__init__.py`

    ```shell
    "ZhinaoForCausalLM": ("zhinao", "ZhinaoForCausalLM"),
    ```

### vLLM Service Start

Start the service:
```shell
python -m vllm.entrypoints.openai.api_server \
    --served-model-name 360Zhinao2-7B-Chat-4K \
    --model qihoo360/360Zhinao2-7B-Chat-4K \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --port 8360
```

Use curl to request the service:
```shell
curl http://localhost:8360/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "360Zhinao2-7B-Chat-4K",
    "max_tokens": 200,
    "top_k": -1,
    "top_p": 0.8,
    "temperature": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好"}
    ],
    "stop": [
        "<eod>",
        "<|im_end|>",
        "<|im_start|>"
    ]
}'
```
Use python to request the service:
```python
from openai import OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8360/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="360Zhinao2-7B-Chat-4K",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好"},
    ],
    stop=[
        "<eod>",
        "<|im_end|>",
        "<|im_start|>"
    ],
    presence_penalty=0.0,
    frequency_penalty=0.0
)
print("Chat response:", chat_response)
```

> If you need to enable repetition penalty, we recommend setting `presence_penalty` and `frequency_penalty` instead of `repetition_penalty`.


<br>

# Model Finetune
## Training data

Training Data: `data/training_data_sample.json`. This example data has 10,000 rows sampled from [multiturn_chat_0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M) with converted format.

Data Format:
```json
[
  {
    "id": 1,
    "conversations": [
        {
            "from": "system",
            "value": "You are a helpful assistant."
        },
        {
            "from": "user",
            "value": "您好啊"
        },
        {
            "from": "assistant",
            "value": "你好！我今天能为您做些什么？有什么问题或需要帮助吗? 我在这里为您提供服务。"
        }
    ]
  }
]
```
## Finetuning scripts
```shell
set -x

HOSTFILE=hostfile
DS_CONFIG=./finetune/ds_config_zero2.json

# PARAMS
LR=5e-6
EPOCHS=3
MAX_LEN=4096
BATCH_SIZE=4
NUM_NODES=1
NUM_GPUS=8
MASTER_PORT=29500

IS_CONCAT=False # Whether to concatenate to maximum length (MAX_LEN)

DATA_PATH="./data/training_data_sample.json"
MODEL_PATH="qihoo360/360Zhinao2-7B-Base"
OUTPUT_DIR="./outputs/"

deepspeed --hostfile ${HOSTFILE} \
        --master_port ${MASTER_PORT} \
        --num_nodes ${NUM_NODES} \
        --num_gpus ${NUM_GPUS} \
        finetune.py \
        --report_to "tensorboard" \
        --data_path ${DATA_PATH} \
        --model_name_or_path ${MODEL_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --model_max_length ${MAX_LEN} \
        --num_train_epochs ${EPOCHS} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps 1 \
        --save_strategy steps \
        --save_steps 200 \
        --learning_rate ${LR} \
        --lr_scheduler_type cosine \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --adam_epsilon 1e-8 \
        --max_grad_norm 1.0 \
        --weight_decay 0.1 \
        --warmup_ratio 0.01 \
        --gradient_checkpointing True \
        --bf16 True \
        --tf32 True \
        --deepspeed ${DS_CONFIG} \
        --is_concat ${IS_CONCAT} \
        --logging_steps 1 \
        --log_on_each_node False
```
```shell
bash finetune/ds_finetune.sh
```
- Configuring `HOSTFILE` switches between single-machine and multi-machine training.
- configuring `ds_config` switches between zero1, zero2 and zero3.
- `fp16, bf16` could configure mixed precision training. bf16 is recommended to be consistent with the pretrained model.
- `is_concat` configures whether the training data is concatenated or not.

<br>

# License

The source code of this repository follows the open-source license Apache 2.0.

360​Zhinao open-source models support free commercial use. It is not necessary for you to submit a request for commercial usage. 
