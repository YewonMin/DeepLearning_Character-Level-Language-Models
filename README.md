# DeepLearning Course - Character-Level Language Models (Shakespeare NN)
This repository contains implementations of **character-level language models** trained on the **Shakespeare dataset**.  
The goal is to compare two recurrent architectures — **Vanilla RNN** and **LSTM** — in their ability to learn long-range dependencies and generate coherent text at the character level.  

Both models are trained on the same dataset and evaluated based on **training/validation loss trends** and text **generation quality**.  


## Folder Structure
```bash
├── dataset.py              # Loads and preprocesses the Shakespeare dataset
├── generate.py             # Generates characters from the best-performing trained model 
├── main.py                 # Trains and evaluates the models; plots train/validation loss
├── model.py                # Defines Vanilla RNN and LSTM models
├── shakespeare_train.txt   # Dataset: Shakespeare’s text corpus
```

## Dataset and Input Processing
The dataset used is Shakespeare’s text corpus (`shakespeare_train.txt`), tokenized at the character level.
Each character is encoded as an integer index, and input–target pairs are created such that the model learns to predict the next character given the current sequence.
Data is split into:
* Training set: 80%
* Validation set: 20%


## Model Configuration
Both RNN and LSTM models were implemented for character-level text prediction.
Each model consists of the following layers:
* Embedding layer: converts character indices into dense vectors.
* Recurrent layer:
  * RNN uses a two-layer nn.RNN.
  * LSTM uses a two-layer nn.LSTM.
* Fully connected layer: projects the hidden outputs to the vocabulary size.


## Training Configuration
| **Parameter**       | **Value**     |
|----------------------|---------------|
| Batch size           | 64            |
| Epochs               | 20            |
| Learning rate        | 0.003         |
| Hidden size          | 128           |
| Number of layers     | 2             |
| Loss function        | CrossEntropyLoss |
| Optimizer            | Adam          |


## Training & Evaluation
### Training Loss & Validation Loss
* #### RNN
![딥러닝_1](https://github.com/YewonMin/DeepLearning_Character-Level-Language-Models/assets/108216502/8ab9aa10-df33-44d2-a44c-c6f5e24cb224)
Both training and validation losses reach their lowest values around **epoch 6**, followed by a slight increase.
* #### LSTM
![딥러닝_2](https://github.com/YewonMin/DeepLearning_Character-Level-Language-Models/assets/108216502/cd8e44ec-aea2-4a53-aa9a-303de7897576)
Both training and validation losses steadily decrease up to **epoch 20**, showing stable convergence.


## Comparison: RNN vs. LSTM
#### Overall Loss Trends (20 Epochs)
![image](https://github.com/YewonMin/DeepLearning_Character-Level-Language-Models/assets/108216502/4ab95fef-8b51-43dc-b8c4-d4136db63fd8)
|   **Model**   |  **Training Loss**    |   **Validation Loss**   |   **Observation**   |
|---------------|-----------------------|-------------------------|---------------------|
|   RNN         |   1.5379 → 1.3978     |   1.4254 → 1.4003       |   Slight decrease, potential overfitting after early epochs   |
|   LSTM        |   1.5019 → 0.9555     |   1.3097 → 0.9848       |   Significant and consistent decrease  |


#### Analysis
* #### Training Loss
  * RNN: Gradually decreases but begins to rise after several epochs — indicates limited capacity to retain long-term dependencies.
  * LSTM: Decreases sharply and remains low, demonstrating more stable learning.

* #### Validation Loss
  * RNN: Initially drops but later increases, suggesting **overfitting**.
  * LSTM: Continues to decline, indicating **better generalization**.

#### Conclusion
- **LSTM outperforms RNN** — lower training and validation losses imply stronger language modeling and generalization capability.
- **RNN shows possible overfitting**, as validation performance plateaus and slightly worsens after early epochs.


## Text Generation with the Best Model
* ##### BEST trained model: LSTM
* ##### Seed characters: `['S', 'T', 'O', 'A', 'G']`
* ##### Sampling Method: Softmax with temperature scaling (`T = 0.5`, `1.0`, `1.5`)

##
#### Temperature Effects in Text Generation
Temperature (`T`) controls the randomness of sampling from the model’s probability distribution:
* T < 1: Sharper distribution → more deterministic, repetitive text
* T = 1: Balanced → natural and fluent text
* T > 1: Softer distribution → diverse but less coherent text
  
##
#### T = 0.5
```bash
Seed: S → "Marcius, He is with your house, my lord..."
Seed: T → "There is the one the death of your swords..."
Seed: O → "ORIOLANUS: Shall I be put upon 't..."
Seed: A → "ANUS: The senate crown'd: your lordship..."
Seed: G → "GLOUCESTER: What then let the people's power..."
```
#### Analysis:
* Text is grammatically stable and coherent, guided by confident predictions.
* Slightly repetitive, limited linguistic diversity.
  
##
#### T = 1.0
```bash
Seed: S → "I'll walls, as more voice, and you should heard..."
Seed: T → "TH: Back out, do not stand upon enge?...”
Seed: O → "OLANUS: What, my gracious lord!...”
Seed: A → "A My holy blind In all our wall'd..."
Seed: G → "GLOUCESTER: The gods have such being answer'd up..."
```
#### Analysis:
* Produces diverse and contextually richer text.
* Occasional grammatical errors, but overall creative and natural.
  
##
#### T = 1.5
```bash
Seed: S → "I know, that prithee; morrow here began's inclicamed..."
Seed: T → "Tower: I not knee heavy; it's good?..."
Seed: O → "OMenes of Towerds fight. Hads IItitud--..."
Seed: A → "ALERIA: No nests great one That stock i' the nobility..."
Seed: G → "GLOLUNipising: whether's patience. Officiaten we haply..."
```
#### Analysis:
* Generates highly varied and unpredictable text.
* Loses grammatical and semantic consistency.
* Demonstrates the trade-off between creativity and coherence.

##
#### Conclusion
* Choosing the appropriate temperature depends on the goal:
  * For coherent text → use T = 0.5
  * For creative and diverse outputs → use T = 1.0 ~ 1.5
* Finding a balance between text quality and variety is crucial in character-level language modeling.


## Key Takeaways
* Demonstrates end-to-end data preprocessing, model training, and evaluation on a text corpus.
* Provides direct comparison between Vanilla RNN and LSTM for character-level generation.
* Highlights how temperature scaling affects output diversity and coherence.
* Shows that LSTM achieves superior performance and more human-like text generation.
