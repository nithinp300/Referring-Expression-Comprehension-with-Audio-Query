# Referring Expression Comprehension with Audio Query

## Abstract
Referring Expression Comprehension (REC) aims to localize a target object in an image based on a referring expression in natural language. While REC typically uses text queries, we propose a model that takes audio queries as input. We leverage the End-to-End TransVG backbone, replacing the text transformer with an audio transformer model. As there are no existing spoken datasets for REC, we introduce a novel audio dataset. We compare text and audio query-based REC models and evaluate various audio transformer models.

## Introduction
Referring expressions play a crucial role in everyday interactions, combining language and vision to define relationships between objects in images. REC remains challenging due to the need for understanding complex language semantics and visual information. Unlike object detection, REC refers to objects using natural language expressions, which can encode detailed information about object instances and their relationships.

## Related Work
Previous work in REC has primarily focused on text queries, with limited research on REC with audio queries. Standard datasets like REFERCOCO provide queries in text format, lacking corresponding spoken queries. We address this gap by generating a synthetic audio dataset using FastSpeech 2.

## Methodology
We propose a REC model that extracts features from audio waveforms using transformer-based models like WAVLM, Wav2Vec2, Unispeech, and Hu-BERT. Visual features are extracted using a convolutional backbone network and visual transformer. The audio and visual representations are concatenated and passed to a Visual-linguistic Fusion Module for final processing.

## Experiments
We compare the performance of audio-based REC models to text-based models in terms of inferencing time and accuracy. Additionally, we evaluate the performance of various audio transformer models to determine their effectiveness in REC tasks.

## Conclusion
Our study demonstrates the feasibility of using audio queries for REC tasks and highlights the importance of considering multi-modal approaches for improved performance. Future research could explore more advanced audio transformer models and further enhance the integration of audio queries in REC.
