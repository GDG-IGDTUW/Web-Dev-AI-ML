# 🫀 Heart Disease Prediction 

Heart disease remains a leading cause of mortality worldwide, and early detection can significantly improve patient outcomes. The Heart Disease Prediction project harnesses the power of Machine Learning to analyze critical health parameters—such as cholesterol levels, blood pressure, and more—to predict the likelihood of heart disease.

This tool is designed to empower individuals and healthcare professionals with actionable insights, facilitating timely intervention and better health management.

Whether you’re a patient monitoring your heart health or a healthcare provider seeking support in decision-making, this application represents a step toward data-driven healthcare solutions.

Experience the intersection of technology and medicine, where AI enhances health awareness and prevention.

# 🙌 Maintainers 👩‍💻 :

- [Vani Varanya](https://github.com/vanivaranya)
- [Saumya Gupta](https://github.com/ISaumya1011)
- [Shruti Narang](https://github.com/Shruti-Narang)

# 🙌 Contributor 👩‍💻 :

- [Saumya Gupta](https://github.com/ISaumya1011)

---

## OVERVIEW:
The Heart Disease Prediction project is a web application designed to predict the likelihood of heart disease in individuals. Leveraging Machine Learning algorithms, the app analyzes user input data—such as age, gender, blood pressure, cholesterol levels, and other relevant health metrics—to provide a predictive assessment. This tool aims to assist users in understanding their heart health and encourages proactive measures by offering insights based on their health parameters.

## Key Features:

 - **User Input Form:** Collects essential health-related information from the user.
 - **Prediction Model:** Utilizes a trained Machine Learning model to analyze input data and predict the probability of heart disease.
 - **Result Display:** Presents the prediction outcome to the user in an understandable format.
 - **Interactive Plots:**: Condition Count by Sex, Boxplot for outliers, Correlation heatmap, etc.
   ![alt text](image.png) ![alt text](image-1.png) ![alt text](image-2.png)


---

## ⚙️ Technical Details:

### Machine Learning Model: 
Trained using relevant datasets to ensure accurate predictions.
### Frontend: 
Simple and user-friendly interface for data input and result presentation.
### Backend: 
Implemented using Flask, a lightweight WSGI web application framework in Python.

---

## Project Category: 
Machine Learning

---
## Workflow

### 1. Importing Necessary Dependencies
The following libraries are used:
- **NumPy & Pandas**: Data handling and numerical operations.
- **Pickle**: Model serialization for future use.
- **Scikit-learn**: Preprocessing, model training, evaluation, and hyperparameter tuning.
- **Matplotlib & Seaborn**: Data visualization for analysis.

### 2. Data Loading and Preprocessing
- The dataset is loaded into a Pandas DataFrame.
- The `condition` column is renamed to `target` for clarity.
- Missing values and outliers are handled appropriately.

### 3. Exploratory Data Analysis (EDA)
- Visualizations like count plots, correlation heatmaps, and boxplots are used to understand feature relationships and outliers.

### 4. Feature Scaling & Splitting Data
- Features (`X`) and target (`y`) variables are separated.
- The dataset is split into training (75%) and testing (25%) sets.
- `StandardScaler` is applied for normalization.

### 5. Model Training & Evaluation
- A **Random Forest Classifier** is trained.
- Predictions are made on the test set.
- Accuracy, classification reports, and confusion matrices are generated for evaluation.

### 6. Hyperparameter Tuning
- `GridSearchCV` is used to find optimal hyperparameters for the Random Forest model.
- The best model is evaluated on the test set.

### 7. Model Serialization
- The trained model is saved as a pickle file (`.pkl`) for future use.

## Contributors
- **Akshay** [github](https://github.com/Akshayk05)
---

## Model Training Process
-**Data Preprocessing:** A dataset containing medical information about patients with heart diseases is uploaded and relevant visualization are plotted. Missing information and outliers in the data are dealt with using statistical methods.

-**Feature Engineering:** The target variable and features are split into training and testing sets, which are scaled to normalize values.

-**Model Training and Evaluation:** A random forest classifier with n_estimators = 20 is trained on the training data. Testing revealed an accuracy of 81.33%.

-**Hyperparameter Tuning for Improved Accuracy:** Grid Search is done to obtain the optimal hyperparameters for the Random Forest Classifier. The accuracy score improved to 86.47%.

-**Model Saving:** The final trained model is saved as a pickle file to be used in the flask application.

---

## 🛠️ How to Get Started  

1. **Fork this Repository**  
   Click the **Fork** button to create your copy of this repository.  

2. **Clone the Repository**  
   ```bash  
   git clone https://github.com/GDG-IGDTUW/Web-Dev-AI-ML.git  
   cd repo-name  
   ```  

3. **Navigate to Project and Setup environment**  
   Navigate to the project folder you're interested in.
   
   ```bash  
   cd Fake-News-Detector
   ```
   Now, after setting path, run following commands on command prompt.
   
   ```bash  
   python -m venv venv
   ```
   
   Followed by
   
    ```bash  
   .\venv\Scripts\activate
   ```

4. **Install Dependencies**
   Load the dataset (if any) and Install necessary Libraries
   
   Install requirements

   ```bash  
   pip install requirements.txt
   ```     

5. **In case of installation error** (Skip step 5 if successful Step 4)

   Install separate dependencies

   For example:
   ```bash  
   pip install "library_name"
   ``` 

7. **Make Your Contributions**  
   - Add Features.
   - Train models.
   - Enhance Accuracy.
   - Improve UI.
   - Test your changes.  

8. **Run and test your changes**  
   Run the Flask Application  
   For example:  
   ```bash  
   python app.py
   ```  

9. **Submit a Pull Request**  
   Push your changes and create a pull request to propose your contributions! 🎉  

---


## 🤝 Contributing Guidelines  

We ❤️ contributions! Follow these simple steps to contribute:  

1. **Browse through Issues and Choose any**  
   Browse the [Issues](#) tab and comment on the one you'd like to work on.  

2. **Clone the Repo, Make changes and Branch Out**  
   Create a new branch for your changes:  
   ```bash  
   git checkout -b feature-name  
   ```  

3. **Commit Your Work**  
   Write clear and concise commit messages:  
   ```bash  
   git commit -m "Add: Feature description"  
   ```  

4. **Push and PR**  
   Push your branch and create a pull request for review.  

---

🌟 Tips for Contributors
 - Follow the repository’s code style and structure.
 - Keep ML scripts well-indented and include comments.
 - Share any interesting results or insights in the pull request description.
 - If you want an issue to be assigned to you, Tag us and mention so under the issue.
 - Please be patient and Feel free to Tag the maintainers or collaborators for any queries. ❤️

---

Happy Coding and Collaborating!🚀❤️
