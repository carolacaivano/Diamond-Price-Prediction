# xtream AI Challenge

## Ready Player 1? 🚀

Hey there! If you're reading this, you've already aced our first screening. Awesome job! 👏👏👏

Welcome to the next level of your journey towards the [xtream](https://xtreamers.io) AI squad. Here's your cool new assignment.

Take your time – you've got **10 days** to show us your magic, starting from when you get this. No rush, work at your pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### What You Need to Do

Think of this as a real-world project. Fork this repo and treat it as if you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the data and craft your own effective solutions.

🚨 **Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* Your understanding of the data
* The clarity and completeness of your findings
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Keep This in Mind**: This isn't about building the fanciest model: we're more interested in your process and thinking.

---

### Diamonds

**Problem type**: Regression

**Dataset description**: [Diamonds Readme](./datasets/diamonds/README.md)

Meet Don Francesco, the mystery-shrouded, fabulously wealthy owner of a jewelry empire. 

He's got an impressive collection of 5000 diamonds and a temperament to match - so let's keep him smiling, shall we? 
In our dataset, you'll find all the glittery details of these gems, from size to sparkle, along with their values 
appraised by an expert. You can assume that the expert's valuations are in line with the real market value of the stones.

#### Challenge 1

Plot twist! The expert who priced these gems has now vanished. 
Francesco needs you to be the new diamond evaluator. 
He's looking for a **model that predicts a gem's worth based on its characteristics**. 
And, because Francesco's clientele is as demanding as he is, he wants the why behind every price tag. 

Create a Jupyter notebook where you develop and evaluate your model.

#### Challenge 2

Good news! Francesco is impressed with the performance of your model. 
Now, he's ready to hire a new expert and expand his diamond database. 

**Develop an automated pipeline** that trains your model with fresh data, 
keeping it as sharp as the diamonds it assesses.

#### Challenge 3

Finally, Francesco wants to bring your brilliance to his business's fingertips. 

**Build a REST API** to integrate your model into a web app, 
making it a cinch for his team to use. 
Keep it developer-friendly – after all, not everyone speaks 'data scientist'!

#### Challenge 4

Your model is doing great, and Francesco wants to make even more money.

The next step is exposing the model to other businesses, but this calls for an upgrade in the training and serving infrastructure.
Using your favorite cloud provider, either AWS, GCP, or Azure, design cloud-based training and serving pipelines.
You should not implement the solution, but you should provide a **detailed explanation** of the architecture and the services you would use, motivating your choices.

So, ready to add some sparkle to this challenge? Let's make these diamonds shine! 🌟💎✨

---

## How to run
#### Challenge 1
Create a conda environment with the following command:
```
$ conda env create -f environment.yml -n diamonds 
```
The notebook you'll be working with is named price_prediction.ipynb, you can find it in the folder solution. To get started, simply activate the environment and initiate the Jupyter Notebook server by executing the following command.

```
$ conda activate diamonds
$ jupyter -notebook
```

#### Challenge 2
In the same conda environment you van execute the script pipeline.py, that can be found in the folder solution. You can run it with the following command:
```
$ python pipeline.py path_of_dataset
```
you need to specify the path where your dataset is located.

#### Challenge 3
In the same conda enviroment you can run the web app with the following command:

```
$ uvicorn api:app --reload --port 8010
```
and navigate the URL http://127.0.0.1:8010/docs, insert your parameters and click execute to get your prediction.


#### Challenge 4
We are going to use google cloud platform, below you can see a representation of the architecture:
![architecture](https://github.com/carolacaivano/xtream-ai-assignment-engineer/blob/main/challenge4.png)

1. Our data can be stored as a CSV file in Google Cloud Storage
2. After uploading our dataset we can use google's cloud manage machine learning platform, VeretexAI. Utilizing Workblench VertexAI, we can create a notebook for data analysis and model training.
3. Once we have trained our model it's essential to integrate it into the Vertex AI Model Registry. This serves as a centralized repository for managing the lifecycle of our machine learning model.
4. Finally we need to deploy our model to an endpoint. Once deployed on Vertex AI, our model functions like any other REST endpoint, accessible for calls from web applications.