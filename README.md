[Jupyter file](https://www.google.com/url?q=https://drive.google.com/file/d/1tvw-udN20U4LQo7mG32s-fRms2CB9oKl/view?usp%3Dsharing&sa=D&source=editors&ust=1774365521455659&usg=AOvVaw2nnH6JzIvncqRmBWn5xc-5)



[PDF](https://www.google.com/url?q=https://drive.google.com/file/d/1UXlEU2-l7wcDYPqkBykyllV1-pYfLsFB/view?usp%3Ddrive_link&sa=D&source=editors&ust=1774365521455404&usg=AOvVaw2KXMHr1S1qyEaVuHh2AILX)

# nga-painting-similarity
 A model to find similarities in paintings, e.g. portraits with a similar face or pose

## Semantic Similarity Engine for Fine Art
This repository contains a multi-modal retrieval system designed to find visually and contextually similar paintings within large art collections. Using the National Gallery of Art (NGA) dataset, the engine combines semantic embeddings, pose detection, and style classification to provide art-historically coherent results.

Architecture
The system uses an adaptive weighted fusion of four distinct components:

 CLIP (ViT-L/14) Backbone: Encodes paintings into 768-dimensional vectors to capture subject matter and mood. It utilizes prompt ensembling (averaging multiple prompts like "an oil painting of...") to bridge the domain gap between natural photography and classical art.

 Portrait Sub-Index: A specialized FAISS index that activates when a query is identified as a portrait. This increases similarity precision for faces and busts from 0.27 to over 0.80.

 Style & Era Classification: Classifies paintings into categories (e.g., Baroque, Renaissance, Impressionist). A "soft penalty" (0.7x multiplier) is applied to results from mismatched eras to maintain historical consistency.

 Pose Detection (MediaPipe): Extracts 33 body keypoints from figures. These coordinates are normalized (re-centered and scaled) and converted into joint angles to ensure pose similarity is invariant to the figure's position in the frame.

## Features
Weighted Fusion: The final similarity score is calculated using (1.0 × CLIP + 0.6 × portrait + 0.5 × pose) / total_active_weights.

Domain Adaptation: No retraining was required; the system uses prompt engineering and normalization to handle the specific nuances of oil paintings.

Automated Gating: The system automatically adjusts weights if a component (like pose detection) does not find a valid target in the image.

## Dataset
This project utilizes the National Gallery of Art collection. Due to size constraints, the image data is not included in this repository.

Source: https://github.com/NationalGalleryOfArt/opendata

Total Images: 4,037


## Usage
Clone the repository.

Install dependencies: 
pip install numpy pandas torch faiss-cpu mediapipe opencv-python.

Open the .ipynb file in Kaggle or a local Jupyter environment.

Point the data directory to the downloaded NGA dataset.
