# InternVL3 NPUQwen3-VL-2B NPU
![Alt text](https://github.com/user-attachments/assets/6d297a34-c516-4cb1-be4a-bca471d40fa6)
<br><br>**User**:\<image\>Describe the image.<br><br>
**Answer**: This is a surreal and imaginative photograph of an astronaut on the Moon, set against the backdrop of Earth in space.

The central figure is an astronaut wearing a white spacesuit with a gold-tinted helmet visor. The suit appears to be made of a light-colored fabric, possibly cotton or a synthetic material like nylon, which has been stretched over a rigid structure. The astronaut's posture is relaxed and casual; they are reclining on the lunar surface, their legs crossed at the ankles.

The astronaut is holding a green glass bottle with a white label that reads "Coca-Cola" in red script. They appear to be drinking from it or have just finished pouring it into a small cup. The astronaut's right hand is resting on the bottle, while their left arm is bent and placed near their head.

In front of the astronaut is a green cooler with a white stripe running along its side. The cooler has a label that reads "Coca-Cola" in red script. It appears to be filled with ice or frozen drinks.

The background features the vastness of space, with stars visible in the dark sky. A large planet, Earth, is prominently displayed behind the astronaut and the Moon's surface. The Earth is shown as a blue and white sphere with swirling clouds and landmasses, partially illuminated by sunlight.

On the right side of the image, there is a small ladder leading up to a structure on the lunar surface. This structure appears to be made of metal or concrete and has some kind of equipment attached to it.

The overall scene combines elements of science fiction and reality, creating an atmosphere that is both humorous and mysterious. The juxtaposition of the astronaut's casual behavior with the vastness of space and Earth creates a sense of wonder and exploration.

------------

## Qwen3-2B VLM for RK3588 NPU (Rock 5, Orange Pi 5). <br/>
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)<br/><br/>
Paper: [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388)<br/><br/>
Hugging face: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

------------

## Introduction

LLMs (Large Language Models) are neural networks trained on large text datasets to understand and generate language.<br>
VLMs (Vision-Language Models) add a visual encoder so the model can process images and text together.<br> 
A combined VLM+LLM system is often referred to as a multimodal model.

These models can be large—hundreds of millions to billions of parameters—which impacts accuracy, memory use, and runtime speed.<br>
On edge devices like the RK3588, available RAM and compute are limited, and even the NPU has strict constraints on supported operations.<br>
Because of this, models typically need to be quantised or simplified to fit.

Performance is usually expressed in tokens (words) per second.<br>
Once converted to RKNN, parts of the model can run on the NPU, improving speed.<br>
Despite these limits, models like Qwen3-2B run well on the RK3588 because the NPU efficiently accelerates the heavy math, and the vision encoder can be optimised. This makes advanced multimodal AI feasible on small, power-efficient devices.

------------

## Model performance benchmark (FPS)

All models, with C++ examples, can be found on the Q-engineering GitHub.<br><br>
All LLM models are quantized to **w8a8**, while the VLM vision encoders use **fp16**.<br>

| model         | RAM (GB)<sup>1</sup> | llm cold sec<sup>2</sup> | llm warm sec<sup>3</sup> | vlm cold sec<sup>2</sup> | vlm warm sec<sup>3</sup> | Resolution | Tokens/s |
| --------------| :--: | :-----: | :-----: | :--------: | :-----: | :--------:  | :--------: |
| [Qwen3-2B](https://github.com/Qengineering/Qwen3-VL-2B-NPU) | 3.1 | 21.9 | 2.6 | 10.0  | 0.9 | 448 x 448 | 11.5 |
| [Qwen3-4B](https://github.com/Qengineering/Qwen3-VL-4B-NPU) | 8.7 | 49.6 | 5.6 | 10.6  | 1.1 | 448 x 448 | 5.7 |
| [Qwen2.5-3B](https://github.com/Qengineering/Qwen2.5-VL-3B-NPU) | 4.8 | 48.3 |  4.0 | 17.9  | 1.8 | 392 x 392 | 7.0 |
| [Qwen2-7B](https://github.com/Qengineering/Qwen2-VL-7B-NPU) | 8.7 | 86.6 |   34.5 | 37.1  | 20.7 | 392 x 392 | 3.7 |
| [Qwen2-2.2B](https://github.com/Qengineering/Qwen2-VL-2B-NPU) | 3.3 | 29.1 |   2.5 | 17.1  | 1.7 | 392 x 392 | 12.5 |
| [InternVL3-1B](https://github.com/Qengineering/InternVL3-NPU) | 1.3 |  6.8 |   1.1 | 7.8    | 0.75 | 448 x 448 | 30 |
| [SmolVLM2-2.2B](https://github.com/Qengineering/SmolVLM2-2B-NPU) | 3.4 | 21.2 |   2.6 | 10.5   | 0.9  | 384 x 384 | 11 |
| [SmolVLM2-500M](https://github.com/Qengineering/SmolVLM2-500M-NPU) | 0.8 |  4.8 |   0.7 | 2.5    | 0.25 | 384 x 384 | 31 |
| [SmolVLM2-256M](https://github.com/Qengineering/SmolVLM2-256M-NPU) | 0.5 |  1.1 |   0.4 | 2.5    | 0.25 | 384 x 384 | 54 |

<sup>1</sup> The total used memory; LLM plus the VLM. <br>
<sup>2</sup> When an llm/vlm model is loaded for the first time from your disk to RAM or NPU, it is called a cold start.<br>
The duration depends on your OS, I/O transfer rate, and memory mapping.<br> 
<sup>3</sup> Subsequent loading (warm start) takes advantage of the already mapped data in RAM. Mostly, only a few pointers need to be restored.<br><br>
<img width="600" height="450" alt="Plot_1" src="https://github.com/user-attachments/assets/6dbb7d8a-c55b-49b4-ae5a-ac272aade60b" /><br>
<img width="600" height="450" alt="Plot_2" src="https://github.com/user-attachments/assets/c4dd409b-2533-4ae0-942c-58fb3e8f5834" />

------------

## Dependencies.
To run the application, you have to:
- OpenCV 64-bit installed.
- rkllm library.
- rknn library.
- Optional: Code::Blocks. (```$ sudo apt-get install codeblocks```)

### Installing the dependencies.
Start with the usual 
```
$ sudo apt-get update 
$ sudo apt-get upgrade
$ sudo apt-get install cmake wget curl
```
#### OpenCV
To install OpenCV on your SBC, follow the Raspberry Pi 4 [guide](https://qengineering.eu/install-opencv-on-raspberry-64-os.html).<br><br>
Or, when you have no intentions to program code:
```
$ sudo apt-get install lib-opencv-dev 
```
------------

## Installing the app.
```
$ git clone https://github.com/Qengineering/Qwen3-VL-2B-NPU
```

#### RKLLM, RKNN
To run InternVL3, you need to have the **rkllm-runtime** library version **1.2.3** (or higher) installed, as well as the **rknpu driver** version **0.9.8**.<br>
If you don't have these on your machine, or if you have a lower version, you need to install them.<br>
We have provided the correct versions in the repo.<br>
```bash
$ cd ./Qwen3-VL-2B-NPU/aarch64/library
$ sudo cp ./*.so /usr/local/lib
$ cd ./Qwen3-VL-2B-NPU/aarch64/include
$ sudo cp ./*.h /usr/local/include
```
### Download the LLM and VLM model.
The next step is downloading the models.<br>
This time, we used the original model supplied by Rockchips [rkllm_model_zoo](https://console.box.lenovo.com/l/l0tXb8) (44 GB!), fetch code: rkllm.<br><br>
Better to download the two needed files (2.3 GB) from our Sync.com server:<br>
[qwen3-vl-2b-instruct_w8a8_rk3588.rkllm](https://ln5.sync.com/dl/6cd2e45d0#swbgmrgn-xqjwb4pn-h3fizzg5-vb3jvfxd) and [qwen3-vl-2b-vision_rk3588.rknn](https://ln5.sync.com/dl/d1a22a380#kshmvhzf-ma8xhheb-mbx2x47f-qp5fajj5)<br>
Copy both into this folder.


### Building the app.
Once you have the two models, it is time to build your application.<br>
You can use **Code::Blocks**.
- Load the project file *.cbp in Code::Blocks.
- Select _Release_, not Debug.
- Compile and run with F9.
- You can alter command line arguments with _Project -> Set programs arguments..._ 

Or use **Cmake**.
```
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
```
### Running the app.
The app has the following arguments.
```bash
VLM_NPU Picture RKNN_model RKLLM_model NewTokens ContextLength
```
| Argument   | Comment |
| --------------| --  |
| picture | The image. Provide a dummy if you don't want to use an image | 
| RKNN_model | The visual encoder model (vlm) | 
| RKLLM_model | The large language model (llm) | 
| NewTokens | This sets the maximum number of new tokens. Optional, default 2048| 
| ContextLength | This specifies the maximum total number of tokens the model can process. Optional, default 4096| 

<br>In the context of the Rockchip RK3588 LLM (Large Language Model) library, the parameters NewTokens and ContextLength both control different limits for text generation, and they're typical in LLM workflows.<br>
**NewTokens**<br> 
This sets the maximum number of tokens (pieces of text, typically sub-word units) that the model is allowed to generate in response to a prompt during a single inference round. For example, if set to 300, the model will not return more than 300 tokens as output, regardless of the prompt length. It's important for controlling generation length to avoid too-short or too-long responses, helping manage resource use and output size.<br>
**ContextLength**<br>
This specifies the maximum total number of tokens the model can process in one go, which includes both the prompt (input) tokens and all generated tokens. For example, if set to 2048 and your prompt already uses 500 tokens, the model can generate up to 2048-500 = 1548 new tokens. This is a hardware and architecture constraint set during model conversion and deployment, as the context window cannot exceed the model's design limit (for instance, 4096 or 8192 tokens depending on the model variant).

A typical command line can be:
```bash
VLM_NPU ./Moon.jpg ./models/qwen3-vl-2b-vision_rk3588.rknn ./models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm 2048 4096
```
The NewTokens (2048) and ContextLength (4096) are optional and can be omitted.
### Using the app.
Using the application is simple. Once you provide the image and the models, you can ask everything you want.<br>
Remember, we are on a bare Rock5C, so don't expect the same quality answers as ChatGPT can provide.<br>
On the other hand, when you see the examples below, the app performs amazingly well.<br><br>
If you want to talk about the picture, you need to include the token `<image>` in your prompt once.<br>
The app remembers the dialogue until you give the token `<clear>`.<br>
With `<exit>`, you leave the application.
### C++ code.  
Below, you find the surprisingly little code of main.cpp. 
```cpp
#include "RK35llm.h"

int main(int argc, char** argv)
{
    std::string input_str;
    std::string output_str;
    RK35llm RKLLM;

    RKLLM.SetInfo(true);            //yes, you may give me additional model information
    RKLLM.SetSilence(false);        //you may print the incremental text chunks on the terminal

    if     (argc< 4) {std::cerr << "Usage: " << argv[0] << " image vlm_model llm_model [option]NewTokens [option]ContextLength\n"; return -1;}
    else if(argc==4) RKLLM.LoadModel(argv[2],argv[3]);
    else if(argc==5) RKLLM.LoadModel(argv[2],argv[3],std::atoi(argv[4]));
    else if(argc> 5) RKLLM.LoadModel(argv[2],argv[3],std::atoi(argv[4]),std::atoi(argv[5]));

    cv::Mat Pic = cv::imread(argv[1]);
    RKLLM.LoadImage(Pic);

    while(true) {
        printf("\n");
        printf("User: ");

        std::getline(std::cin, input_str);
        if (input_str == "exit") break;

        output_str = RKLLM.Ask(input_str);
//        std::cout << "\nLLM Reply: " << output_str << std::endl;
    }

    return 0;
}
```
Most code speaks for itself. One remark.<br>
The LLM generates the answer to your request in little pieces of text. You see them as if the LLM is typing on your terminal.<br>
RKLLM.SetSilence controls this behaviour. When set to `true`, it will suppress this output. RKLLM.Ask() also returns the complete answer.<br>
In main.cpp uncomment this line:
```cpp
std::cout << "\nLLM Answer: " << output_str << std::endl;
```

------------

## VLM examples.
| Image         | Dialog |
| --------------| --  |
| ![Singapore_S](https://github.com/user-attachments/assets/e742059a-3bdb-41be-9647-e7767da12ce6) |**User**: \<image\>Which city is this?<br>**Answer**: This is Singapore, a city in Southeast Asia known for its modern architecture and vibrant culture. It's located on the island of Singapore, which is part of Malaysia but has been an independent nation since 1965.<br><br>The image shows several iconic landmarks:<br>- The Marina Bay Sands hotel with its distinctive "sky garden" design.<br>- The Singapore Flyer, a large observation wheel.<br>- The city's skyline at dusk, reflecting beautifully on the water. | 
| ![ChineseWall_S](https://github.com/user-attachments/assets/3ba4b57a-4b22-4050-b995-9599c6d239de) |**User**: \<image\>Tell me something about the history of this place.<br>**Answer**: The image shows the **Great Wall of China**, one of the most iconic and historically significant structures in human civilization.<br><br>Here is some information about its history:<br><br>- The Great Wall was built over several centuries by various Chinese dynasties to protect against invasions from nomadic groups, particularly the Mongols.<br>- Construction began during the **Qin Dynasty** (221–206 BCE), but it wasn't until the **Ming Dynasty** (1368–1644) that the wall was expanded and fortified in a massive scale. The Ming builders used advanced techniques to construct walls, towers, and watchtowers.<br>- The Great Wall stretches over 13,000 miles (21,000 kilometers), with many sections built on different dates and by different dynasties.<br>- It is not just a wall but also a symbol of Chinese engineering, military strategy, and cultural identity. Today, it is one of the most visited tourist attractions in China.<br><br>The Great Wall has been an important part of Chinese history for over 2,000 years and continues to be a source of pride for the Chinese people.|
| ![Pizza_S](https://github.com/user-attachments/assets/26fd5da3-f491-4b48-978a-14eae0819247) |**User**: \<image\>What are the ingredients I need to make this?<br>**Answer**: Based on the image provided, this appears to be a classic **Neapolitan-style pizza** or a similar style of **pizza al taglio**, which is known for its simple and authentic ingredients.<br><br>The key components visible in the picture are:<br>-   **Pizza Base:** A round, thin-crust base made from dough.<br>-   **Tomato Sauce:** The red sauce that forms the foundation of the pizza.<br>-   **Cheese:** Typically mozzarella cheese, which is melted over the top.<br>-   **Fresh Basil Leaves:** Used as a garnish and for flavor.<br><br>The most important ingredient to note is the **crust**. This type of crust is often made from a special dough that's traditionally fermented in a wooden oven (a "forno" or "forno di legno") and has a very thin, delicate texture. The pizza is then cooked at high temperatures until it becomes golden brown.<br><br>The ingredients you would need to make this are:<br>-   **Dough:** A simple dough made from flour, water, salt, and sometimes yeast.<br>-   **Tomato Sauce:** A basic tomato sauce made with fresh tomatoes, garlic, olive oil, and seasonings.<br>-   **Mozzarella Cheese:** Fresh mozzarella cheese is typically used for this type of pizza.<br>-   **Fresh Basil Leaves:** For garnish and flavor.<br><br>The key difference from a typical pizza is the use of a very thin crust that's cooked in a wood-fired oven. This gives it a unique texture and flavor, which is why it's often considered a specialty item at Italian restaurants.<br><br>**User**: What is the first step?<br>**Answer**:Based on the image provided, the first step to make this pizza is to **prepare the dough**.<br><br>Here’s a breakdown of what you need to do:<br>1.  **Prepare the Dough:** You will need to mix flour, water, salt, and sometimes yeast to create a soft, pliable dough.<br>2.  **Knead the Dough:** Knead the dough until it becomes smooth and elastic.<br>3.  **Let It Rest:** Let the dough rest for about an hour or two in a warm place.<br>4.  **Shape the Dough:** After resting, shape the dough into a ball and let it rise again for another hour or so.<br><br>Once you have prepared the dough, you can proceed to make the pizza by spreading tomato sauce on top of the dough, adding mozzarella cheese, and finally placing fresh basil leaves on top.|
| ![Titanic_S](https://github.com/user-attachments/assets/bf6da76c-c9a4-41c6-bea1-2b4411dd4013) |**User**: \<image\>Which movie is this?<br>**Answer**: This is the 2008 film *Titanic*, directed by James Cameron. It's an epic romantic drama that tells the story of a young woman, Rose, and her love for Jack Dawson on the ill-fated ship Titanic during its maiden voyage in 1912.<br><br>The image shows Leonardo DiCaprio as Jack Dawson and Kate Winslet as Rose DeWitt, both iconic characters from the film. The scene is set against a dramatic sunset over the ocean, capturing their emotional journey and the tragic fate of the ship.|

## LLM example.
Notice the nice Markdown makeup of the answer!<br><br>
**User**: Can you make me a C++ call that embeds the Pythagorean theorem?<br><br>
**Answer**: Certainly! Here's a **C++** program that uses the **Pythagorean Theorem** to calculate the length of the hypotenuse in a right triangle given the lengths of the two legs. It includes a simple function to compute it using the formula:

\[
c = \sqrt{a^2 + b^2}
\]

We'll use `std::sqrt` from `<cmath>` and define a C++ function that takes two floating-point numbers as input (the two legs) and returns the hypotenuse.

```cpp
#include <iostream>
#include <cmath>
using namespace std;

// Function to calculate the hypotenuse using Pythagorean Theorem
double calculateHypotenuse(double a, double b) {
    return sqrt(a * a + b * b);
}

int main() {
    // Example usage: Calculate hypotenuse for sides 3 and 4
    double leg1 = 3.0;
    double leg2 = 4.0;

    double hypotenuse = calculateHypotenuse(leg1, leg2);

    cout << "The hypotenuse of a right triangle with legs " 
         << leg1 << " and " << leg2 << " is: " << hypotenuse << endl;

    return 0;
}
```

### Output:
```
The hypotenuse of a right triangle with legs 3 and 4 is: 5.0
```

---

### How It Works:

- `calculateHypotenuse(a, b)` computes \( \sqrt{a^2 + b^2} \)
- The program uses the **Pythagorean Theorem** to find the hypotenuse.
- This is a **pure C++ implementation**, not involving Python or any external libraries.

Let me know if you'd like this version to work with **user input**, **multiple test cases**, or even **include error checking**!

------------

[![paypal](https://qengineering.eu/images/TipJarSmall4.png)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=CPZTM5BB3FCYL) 




