# assessment_of_damaged_buildings_in_satellite_images

## Project Overview

This project presents an end-to-end deep learning pipeline for automatically identifying buildings and assessing their damage severity from post-disaster satellite imagery. The solution leverages semantic segmentation and image classification models to detect structures and classify their damage into categories: no damage, minor, major, or destroyed.

The pipeline is designed to support rapid disaster response, aid assessment, and post-event urban planning by providing a visual damage map for affected regions.

## Key Objectives

- Segment building footprints from pre-disaster satellite images using a semantic segmentation model.
- Classify the severity of damage to each detected building using post-disaster imagery.
- Produce a visual overlay highlighting each building with a color-coded indicator of damage level.

## Technical Details

### 1. Building Localization (Segmentation)

- **Model Used:** DeepLabv3+ with ResNet101 backbone
- **Input:** Pre-disaster RGB satellite images
- **Labels:** Binary masks generated from WKT polygons representing building footprints
- **Output:** Pixel-wise building segmentation maps for unseen post-disaster images
- **Implementation:** Custom Keras training pipeline with online data augmentation and sparse categorical loss

### 2. Damage Classification

- **Model Used:** EfficientNet-B0
- **Input:** Cropped building patches extracted from post-disaster satellite images
- **Labels:** Multi-class labels (no-damage, minor-damage, major-damage, destroyed)
- **Output:** Single-label prediction per building patch


### 3. Data Preparation

- Pre-disaster images paired with binary masks created from geospatial WKT polygon labels
- Post-disaster images aligned using disaster metadata
- Dataset includes over 1,500 usable samples after filtering out invalid or empty polygons

### 4. Inference Pipeline

- Takes a post-disaster satellite image as input
- Performs segmentation to locate building regions
- Crops and classifies each detected building
- Generates an Croped images with labeling the Damaged severity level

## Dataset

The dataset used in this project is based on the publicly available xBD (eXtreme Building Damage) dataset, which includes georeferenced satellite imagery and damage annotations in the form of WKT polygons. A subset was used for model training and evaluation after pre-processing to match segmentation and classification requirements.

## Applications

- Disaster response and emergency planning
- Urban damage analysis and resilience studies
- Insurance assessment and risk modeling
- Remote sensing-based infrastructure monitoring





