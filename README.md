# xtream AI Challenge - Software Engineer

## Ready Player 1? 🚀

Hey there! Congrats on crushing our first screening! 🎉 You're off to a fantastic start!

Welcome to the next level of your journey to join the [xtream](https://xtreamers.io) AI squad. Here's your next mission.

You will face 4 challenges. **Don't stress about doing them all**. Just dive into the ones that spark your interest or that you feel confident about. Let your talents shine bright! ✨

This assignment is designed to test your skills in engineering and software development. You **will not need to design or develop models**. Someone has already done that for you. 

You've got **7 days** to show us your magic, starting now. No rush—work at your own pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### Your Mission
[comment]: # (Well, well, well. Nice to see you around! You found an Easter Egg! Put the picture of an iguana at the beginning of the "How to Run" section, just to let us know. And have fun with the challenges! 🦎)

Think of this as a real-world project. Fork this repo and treat it like you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

**Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the problem and craft your own effective solutions.

---

### Context

Marta, a data scientist at xtream, has been working on a project for a client. She's been doing a great job, but she's got a lot on her plate. So, she's asked you to help her out with this project.

Marta has given you a notebook with the work she's done so far and a dataset to work with. You can find both in this repository.
You can also find a copy of the notebook on Google Colab [here](https://colab.research.google.com/drive/1ZUg5sAj-nW0k3E5fEcDuDBdQF-IhTQrd?usp=sharing).

The model is good enough; now it's time to build the supporting infrastructure.

### Challenge 1

**Develop an automated pipeline** that trains your model with fresh data, keeping it as sharp as the diamonds it processes. 
Pick the best linear model: do not worry about the xgboost model or hyperparameter tuning. 
Maintain a history of all the models you train and save the performance metrics of each one.

### Challenge 2

Level up! Now you need to support **both models** that Marta has developed: the linear regression and the XGBoost with hyperparameter optimization. 
Be careful. 
In the near future, you may want to include more models, so make sure your pipeline is flexible enough to handle that.

### Challenge 3

Build a **REST API** to integrate your model into a web app, making it a breeze for the team to use. Keep it developer-friendly – not everyone speaks 'data scientist'! 
Your API should support two use cases:
1. Predict the value of a diamond.
2. Given the features of a diamond, return n samples from the training dataset with the same cut, color, and clarity, and the most similar weight.

### Challenge 4

Observability is key. Save every request and response made to the APIs to a **proper database**.

---

## How to run

---

# Automated Model Training Pipeline

This project provides an automated pipeline for training machine learning models. The pipeline monitors a specific folder for new files, automatically adds them to the main dataset, and retrains existing models.

## Project Structure

- `./test/Folder_Scanner.py`: Main script that monitors a folder for new files and manages the dataset and model retraining.
- `./history/current_models/`: Folder where machine learning models are saved as pickle files.
- `./data/`: Folder containing the main dataset.
- `./test/app.py`: Script to start the project's APIs.

## Requirements

Make sure you have installed the libraries listed in the `requirements.txt` file.

You can install them by running:
```bash
pip install -r requirements.txt
```

## Usage Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LeoBello00/xtream-ai-assignment-developer.git
   cd xtream-ai-assignment-developer
   ```

2. **Navigate to the project folder:**
   Ensure you are in the project's main folder.

3. **Start the `Folder_Scanner.py` script:**
   Run the following command to start monitoring the folder:
   ```bash
   python ./test/Folder_Scanner.py
   ```

4. **Add a New File:**
   When a new file is added to the monitored folder, it will automatically be added to the main dataset and all models saved as pickle files in the `./history/current_models/` folder will be retrained.

5. **Add a New Model:**
   To add a new model to the pipeline:
   - Create and save the model as a pickle file in the `./history/current_models/` folder.
   - The new model will be automatically integrated into the retraining pipeline.

## API Usage

The project's APIs are managed by the `app.py` script located in `./test/app.py`. To start the API server, run the command:

```bash
python ./test/app.py
```

The following endpoints are available:

### **GET /predict**
Returns predictions based on the provided parameters.

**Request Parameters:**
- `color`: Diamond color
- `cut`: Diamond cut
- `carat`: Diamond carat weight (float)
- `depth`: Diamond depth (float)
- `table`: Diamond table (float)
- `clarity`: Diamond clarity
- `x`: Diamond x dimension (float)
- `y`: Diamond y dimension (float)
- `z`: Diamond z dimension (float)

**Example Request:**
```bash
curl -X GET "http://localhost:5000/predict?color=E&cut=Ideal&carat=0.3&depth=61.5&table=55&clarity=VS2&x=4.3&y=4.35&z=2.7"
```

### **GET /similar_samples**
Returns similar samples based on the provided parameters.

**Request Parameters:**
- `n`: Number of similar samples to return
- `cut`: Diamond cut
- `color`: Diamond color
- `clarity`: Diamond clarity
- `weight`: Diamond weight (float)

**Example Request:**
```bash
curl -X GET "http://localhost:5000/similar_samples?n=5&cut=Ideal&color=E&clarity=VS2&weight=0.3"
```

---

