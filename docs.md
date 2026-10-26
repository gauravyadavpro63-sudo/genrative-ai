 Machine Learning vs Deep Learning vs Generative AI

1. Machine Learning (ML)
Teaching computers to learn patterns from data.

Instead of manually writing every rule, you give a model data so it can learn patterns and make predictions.

Example: Give a model thousands of emails labeled spam or not spam. It learns to classify new emails.

Other examples: house-price prediction, fraud detection, recommendation systems.

2. Deep Learning (DL)
A type of machine learning that uses deep neural networks.

These networks contain multiple layers that learn increasingly complex patterns from data.

Example: Train a neural network to recognize cats in photographs. It can learn visual features such as edges, shapes, ears, and faces.

Other examples: speech recognition, image recognition, autonomous driving.

3. Generative AI (GenAI)
AI that generates new content based on what it has learned.

It can produce text, code, images, audio, video, and more.

Example: Ask ChatGPT to write a C++ solution, explain RAG, or generate an image. The model produces a response based on your prompt.

Other examples: ChatGPT, Claude, image-generation models, AI coding assistants.



2. How are they related?
Think of it as a family tree, Kaneki.
Artificial Intelligence (AI)

The broad field of making machines perform intelligent tasks

Machine Learning (ML)

Learns patterns from data

Deep Learning (DL)

Learns using multilayer neural networks

Generative AI

A capability that many deep-learning models provide


Important nuance: This diagram shows a common learning path, not a strict hierarchy. Generative AI is a category defined by what a model does, and it is not always deep learning. Most modern, powerful GenAI systems, however, are built using deep learning.
3. One practical example that makes everything click
Imagine you're building an AI feature for your BITGODS coding platform.
Technology	What you could build
Machine Learning	Predict whether a coding submission might be incorrect based on learned patterns
Deep Learning	Train a neural network to classify code or understand complex code patterns
Generative AI	Generate code explanations, hints, debugging suggestions, or solutions
Agentic AI	Let an AI agent inspect code, run tests, diagnose errors, and try fixes using tools

Notice that Agentic AI is different again: it focuses on taking actions toward a goal, often by using a GenAI model together with tools and a workflow.



What is LLM

LLM stands for Large Language Model.

It's an AI model trained on enormous amounts of text so it can understand and generate human-like language.


Token and tokenization


1. What is a token?
A token is a small unit of text that an LLM processes. It can be a complete word, part of a word, punctuation, or sometimes a single character.

For example, imagine you send this sentence to an LLM:

I love learning AI!

A tokenizer might split it into:
I
 love
 learning
 AI
!

Illustrative example: 5 tokens. Actual tokenization depends on the model's tokenizer.

Notice that a token doesn't have to be a complete word. A long or uncommon word might be split into several smaller pieces.
2. What is tokenization?
Tokenization is the process of splitting text into tokens that a model can process.

For example:
Input text

I love programming!

Tokenizer

Splits text using the model's vocabulary and tokenization rules.

I
 love
 programming
!

Example token pieces

Token IDs

Each token maps to a numerical ID that the model can use.


Why token IDs? Because neural networks work with numerical representations, not raw words directly.

For example, a tokenizer might map tokens to IDs like this:
Token	Illustrative ID
I	40
love	891
AI	5621
!	0

These IDs are made up for explanation. Real IDs depend on the tokenizer.
3. Why do LLMs need tokens?
There are three main reasons:
- Processing text: Tokenization converts text into a form the model can process.
- Generating responses: The model predicts the next token repeatedly to generate an answer.
- Managing cost and context: Many LLM APIs measure input and output usage in tokens. Context windows also have token limits.
For example, when you ask ChatGPT, "What is Generative AI?", the model processes your prompt as tokens and generates a response token by token.



1. What is a Neural Network?
A neural network is a machine-learning model that learns patterns from data by adjusting numerical values called weights.

It's loosely inspired by how neurons connect in the human brain, but artificial neural networks work through mathematics, not like a biological brain.

Imagine you're building an AI that predicts whether a student will pass an exam based on study hours.
Input layer

Study hours: 6

Hidden layer
●
●
●

Neurons perform mathematical calculations

Output layer

Predicted probability of passing: 85%

Illustrative example — not a real prediction.

The network receives input, processes it through layers of artificial neurons, and produces an output.
How does it learn?
Suppose the network makes a bad prediction.
1. It makes a prediction.
2. A loss function measures how far the prediction is from the correct answer.
3. Backpropagation calculates how the weights contributed to the error.
4. An optimizer adjusts those weights.
5. The process repeats across training examples.
Over time, the network can learn useful patterns.
How does this connect to LLMs?
LLMs are built using neural networks, most commonly Transformer architectures.

When you send a message to an LLM:
- Your text is split into tokens.
- Token IDs are converted into numerical vectors called embeddings.
- Transformer layers process those vectors using mechanisms such as attention.
- The model calculates probabilities for possible next tokens.
- Tokens are generated repeatedly to form a response.
So remember:

LLM = a language model built using a neural network, typically a Transformer.

2. What is a Context Window?
A context window is the amount of information an LLM can take into account in a single processing context, measured in tokens.

Imagine you're chatting with an AI while studying Generative AI.

You tell it:
- Your programming background.
- What you've learned about tokens.
- Your current question.
- A long piece of code you want it to debug.
The model needs to process relevant conversation history and your new message to answer appropriately. The context window limits how much information can fit into that context.
Imagine a context window of 10,000 tokens
Illustrative example, not a specification for any particular model.
Previous conversation

4,000 tokens




Current question and code

2,000 tokens




Available space remaining

4,000 tokens





When the available context is exhausted, the application may need to remove or summarize older information, truncate content, or start a new context. The exact behavior depends on the model and application.

A few important details:
- The context window is measured in tokens, not words or characters.
- It can include input messages, conversation history, instructions, and other supplied content.
- Depending on the model, the output may also count against a combined input/output limit.
- A larger context window lets a model handle longer documents or conversations, but it doesn't guarantee perfect recall or attention to every detail.
Context window vs. token limit
These terms are related, but not always identical.
Term	Meaning
Token	A piece of text processed by the model
Tokenization	Converting text into token pieces
Context window	How much information the model can process in context
Output token limit	Maximum tokens allowed in the generated response
Parameters	Learned numerical values inside the model



1. What is a parameter in AI?
A parameter is a numerical value inside a neural network that the model learns during training.

These values help the network recognize patterns and produce predictions.

Think of a parameter as an adjustable number inside the model. During training, the model adjusts many of these numbers to improve its predictions.
2. Simple example
Imagine we're building a model that predicts a student's marks based on study hours.

Suppose our model uses this formula:
\[
\text{Predicted marks} = w \times \text{Study hours} + b
\]

Here:
- \(w\) = weight (a parameter)
- \(b\) = bias (another parameter)
Let's say the model learns that:
- \(w = 10\)
- \(b = 5\)
If a student studies for 6 hours:
\[
\text{Marks} = 10 \times 6 + 5 = 65
\]

Try changing the parameters
Study hours6

1
5
10





Weight (w)10

1
10
20





Bias (b)5

0
10
20






Predicted marks in our toy model
65
Illustrative formula only. Real exam performance is much more complicated.

Notice that changing the parameters changes the prediction. In real neural networks, training adjusts huge numbers of parameters based on errors across training examples.
3. What about parameters in an LLM?
An LLM has many layers of mathematical operations, each containing learned parameters such as weights and biases.

For example, the GPT family, Llama, and other Transformer models have parameters that help them learn language patterns, relationships between words, coding structures, and much more.

You might hear people talk about:
- 7B parameters = 7 billion parameters.
- 70B parameters = 70 billion parameters.
- 400B parameters = 400 billion parameters.
The B means billion.

More parameters generally mean a model has more learned numerical values, but more parameters do not automatically mean a smarter or better model. Training quality, architecture, data, and inference methods also matter.
4. Parameters vs. tokens vs. context window
Concept	Think of it as
Parameters	The model's learned numerical settings
Tokens	Pieces of text the model processes
Tokenization	Splitting text into tokens
Context window	The amount of information the model can process in context
Training	The process through which the model learns its parameters

Here's a useful analogy: parameters are like the knowledge encoded in the model's learned settings, while tokens are the pieces of text it reads and generates. The context window determines how much information it can consider at once.

One subtle point: parameters aren't a literal database of facts. They're numerical values learned during training, and the model uses them to generate outputs.
5. How does an LLM learn its parameters?
The simplified process is:
1. The model receives training text.
2. It predicts the next token.
3. A loss function measures how wrong its prediction was.
4. Backpropagation calculates how the parameters contributed to that error.
5. An optimizer updates the parameters.
6. This repeats across many training examples.


Attention

1. What is attention?
Attention is a mechanism that helps a model determine which other tokens are relevant to a token when processing a sequence.

Imagine I give you this sentence:
"The cat sat on the mat because it was tired."

What does "it" refer to? The cat!

Humans figure this out naturally. An AI model needs a mechanism to learn relationships between tokens. Attention helps it do that.


Example: token relationships
"The cat sat on the mat because it was tired."

"it" → "cat"
The model can learn a strong relationship between these tokens in context.


1. What are embeddings?
Embeddings are numerical vectors that represent tokens, words, or other data in a form a neural network can process.

Computers don't directly understand the meaning of the word "cat". An LLM converts its token ID into a list of numbers called an embedding.

For example, imagine this sentence:

The cat eats fish

The tokenizer converts it into token IDs, and an embedding layer maps those IDs to vectors.
Text
"cat"

Token ID
9246 (illustrative)

Embedding vector
0.21

-0.48

0.73

0.12

-0.36


2. Why does an LLM need embeddings?
Imagine these three words:
- Cat 🐈
- Dog 🐕
- Car 🚗
A well-trained embedding system may represent cat and dog with similar semantic features because both relate to animals. Car may have different features because it relates to vehicles.


3. How do embeddings and attention work together?
This is the important connection, Kaneki. 🧠
Text

"The cat eats fish"

Tokenization
|
Text → token IDs
|
Embeddings
|
Token IDs → vectors
|
Attention
|
Mixes information across relevant tokens
|
Further neural network layers
|
Build richer representations used to predict the next token



Embeddings = represent the tokens as numericals vectors.


Attention = help the model determine which tokens are relevant to one another.



