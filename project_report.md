# Crop Leaf Disease Identifier Using MobileNet

## Title
Crop Leaf Disease Identifier Using MobileNet

## Abstract
Crop diseases are a major cause of reduced yield and economic loss in agriculture. Early and accurate identification of diseased leaves is essential for timely intervention and effective crop management. This project presents a lightweight web-based disease detection system that allows farmers and agronomists to upload a crop leaf image and receive a disease diagnosis along with a recommended treatment plan. The application uses a MobileNet-inspired classification workflow, a Flask backend for image processing and API handling, and a SQLite database for storing prediction records and uploaded images. The system is designed to be simple, low-cost, and suitable for deployment in rural and resource-limited agricultural environments.

## Keywords
Crop disease detection, MobileNet, image classification, Flask, precision agriculture, plant health monitoring

## 1. Introduction
Agriculture plays a vital role in food security and rural livelihoods. However, crop health is often compromised by diseases that can spread quickly if not identified in time. In many cases, farmers rely on visual inspection, which can be inaccurate, delayed, and dependent on expert knowledge. This creates a need for accessible tools that can support early diagnosis and reduce crop losses.

The advancement of deep learning and mobile-friendly neural networks has made it possible to build practical image-based disease detection systems. MobileNet is particularly useful for this purpose because it offers a compact architecture with relatively low computational cost, making it suitable for browser and local deployment scenarios. This project applies that concept in a web application where a user can upload an image of a leaf and receive disease identification results with treatment guidance.

## 2. Problem Statement
Farmers often encounter disease symptoms too late to contain the spread effectively. In many agricultural regions, access to trained agronomists and laboratory testing is limited. As a result, crop health decisions are delayed, leading to yield loss, excessive pesticide usage, and poor field management. A quick and low-cost decision-support system is therefore necessary to detect common plant diseases at an early stage.

## 3. Objectives
The main objectives of this project are as follows:

- To design a lightweight disease detection system for crop leaves.
- To implement a MobileNet-inspired image classification model.
- To build a Flask-based web application for image upload and prediction.
- To provide disease-specific treatment recommendations.
- To store prediction data and uploaded images for review and analysis.
- To prepare a clear and professional documentation suitable for academic submission.

## 4. Motivation
The motivation behind this project is to make disease diagnosis accessible, affordable, and efficient. Many farmers do not have direct access to expert disease analysis, and automated monitoring systems are often expensive or difficult to deploy in small-scale agriculture. This system aims to bridge that gap by delivering a practical, browser-based tool that helps users detect early signs of disease and take action promptly.

## 5. Literature Review and Background
Plant pathology and computer vision have long been used together for disease diagnosis. Traditional methods rely on manual identification by agronomists, which is slow and can be inconsistent. Recent research has shown that deep learning models can recognize leaf disease patterns with good accuracy when trained on large datasets.

MobileNet architectures are widely used because they reduce model size and computational complexity without sacrificing classification performance in many image-processing tasks. Their efficient design makes them a suitable choice for agricultural applications that may run on local machines or low-power systems. This project adopts a similar mobile-first design to create a lightweight plant disease recognition platform.

## 6. System Architecture
The proposed system is composed of four major components:

1. Frontend interface for image upload and result display.
2. Flask backend for request handling and validation.
3. MobileNet-inspired classification layer for prediction.
4. SQLite storage module for saving prediction history and images.

The overall workflow is as follows:

1. The user uploads a leaf image through the browser.
2. The Flask application receives the file and validates it.
3. The image is resized and normalized for model input.
4. The disease classifier predicts the label and confidence score.
5. The system identifies the corresponding treatment advice.
6. The result is displayed to the user and recorded in the database.

## 7. Methodology
The methodology follows a simple yet effective pipeline:

- data preparation and image preprocessing,
- construction of a lightweight CNN-based model,
- generation of synthetic disease patterns for demonstration and training,
- classification of uploaded leaf images,
- treatment mapping based on the predicted disease class,
- result recording in SQLite for traceability.

The classification model is built using a compact deep learning structure inspired by MobileNet principles. The model is trained on representative synthetic patterns across multiple disease classes and then used for inference during prediction. While the system is designed as a practical prototype, it also supports future extension toward real public datasets such as PlantVillage.

## 8. Disease Classes Covered
The application identifies the following disease classes:

- Healthy
- Early Blight
- Late Blight
- Leaf Spot
- Rust
- Powdery Mildew
- Mosaic Virus
- Nutrient Deficiency

These classes were selected to represent common crop health conditions, including fungal infections, viral symptoms, and nutrient-related abnormalities.

## 9. Treatment Recommendation Module
An important feature of the system is its treatment support. After a disease is predicted, the application maps the result to a recommended field response such as fungicide application, pruning, proper irrigation management, nutrient balancing, or vector control. This gives the system more practical value than mere classification and helps users act on predictions in the field.

## 10. Software and Tools Used
The project was implemented using the following technology stack:

- Python 3.10+
- Flask web framework
- TensorFlow / Keras
- NumPy
- Pillow
- SQLite
- HTML, CSS, and JavaScript

These tools were chosen to ensure the project remains lightweight, easy to run, and suitable for academic and demonstration purposes.

## 11. System Design and Implementation
The frontend was designed as a clean dashboard-style interface where users can upload an image and see the disease result immediately. The backend handles file validation, pre-processing, model prediction, treatment mapping, and database insertion. The database stores prediction metadata, including the image path, disease label, confidence score, treatment, and timestamp.

This modular design allows the project to be easily extended. For example, the model layer can be replaced with a larger transfer-learning model trained on a real dataset, while the frontend and database structure remain unchanged.

## 12. Results and Discussion
The implemented prototype successfully demonstrates the core functionality of automatic disease recognition. It accepts an uploaded leaf image, classifies the suspected disease, calculates a confidence value, displays the result, and provides a treatment suggestion. Because it is lightweight and local, the application is accessible without heavy cloud infrastructure and is useful in scenarios where internet connectivity is limited.

The main advantage of the system lies in its simplicity and usability. It can serve as a practical decision-support tool for educational purposes, field demonstrations, and early-stage agricultural technology projects. The solution also lays the groundwork for future extension with larger agricultural datasets and a more advanced model pipeline.

## 13. Limitations
Although the project is effective as a prototype, it has certain limitations. The current model is trained on synthetic patterns rather than a fully real agricultural image dataset. This limits the generalizability of the system for real-world field conditions such as varying lighting, leaf angles, complex backgrounds, and mixed disease symptoms. In addition, disease classes are limited to those defined in the prototype and can be expanded further.

## 14. Future Scope
The project can be expanded in several ways:

- Train the model using a public dataset such as PlantVillage.
- Add more crop species and diseases.
- Integrate a farmer dashboard with analytics and historical trends.
- Improve accuracy using data augmentation and transfer learning.
- Add multilingual support and voice-based guidance for field use.
- Deploy the application in a cloud or mobile environment for broader availability.

## 15. Conclusion
This project demonstrates how deep learning, web development, and agricultural decision support can be combined into a practical system for crop leaf disease identification. By simplifying the diagnosis pipeline and providing treatment guidance, the application supports early detection and encourages better crop management. The system acts as a strong foundation for future agricultural AI solutions and can be extended with larger real-world datasets and more advanced models.

## 16. References
1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., and Chen, L.-C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. IEEE Conference on Computer Vision and Pattern Recognition.
2. TensorFlow Core. TensorFlow documentation and Keras API reference.
3. Flask documentation. Pallets Project.
4. SQLite documentation. SQLite Consortium.
5. PlantVillage dataset. Penn State University and related agricultural image repositories.
6. Plant pathology and precision agriculture literature on disease identification and field monitoring.
