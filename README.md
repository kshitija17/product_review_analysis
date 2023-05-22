
# Analyzing User reviews on Natural Skincare Products through Sentiment Analysis

This project focuses on conducting sentiment analysis on product reviews from natural skincare brands. The objective is to offer valuable insights into customer opinions, attitudes, and preferences, enabling businesses to gain a deeper understanding of how customers perceive natural products.

## Description
Reviews play a crucial role in helping businesses identify opportunities for improvement and gaining insights into the sales and popularity of their products among consumers. Both positive and negative reviews contribute to forecasting product sales and facilitating efficient inventory management. When entering the natural skincare industry, companies can greatly benefit from understanding customer opinions about these products. Moreover, such analysis has the potential to influence customers to choose natural products.

Sentiment analysis involves several stages, which encompass collecting data, preprocessing, selecting features, extracting features, and ultimately determining the sentiment expressed in the reviews. 

## Project Phases
1. The reviews were scrapped from Amazon.in from renowned cosmetics brands using a tool called DataMiner which is a free web scraping tool.
2. Data preprocessing was the pivotal stage in the project. Data preprocessing was performed using Pandas and NLTK, involving the elimination of stopwords, punctuations, HTML tags, URLs, whitespaces, duplicate words, frequent words, rare words, and numbers from the data. Furthermore, tokenization and lemmatization techniques were applied.
3. Before being written in an object-oriented programming format, all model building and testing were conducted in Colab Laboratory.The various machine learning models classified the reviews into positive, neutral, and negative classes. 
4. The complete application was created by utilizing Flask for the backend, enabling the development of a REST API, and React for the front end.
5. The project was containerized using Docker, and subsequently deployed on an Amazon EC2 server.


## Software and Tool requirements
1. [Anaconda](https://www.anaconda.com/)
2. [VSCodeIDE](https://code.visualstudio.com/)
3. [GitCLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)
4. [Github](https://github.com/)
5. [Python 3.11.3](https://www.python.org/downloads/)
6. [scikit-learn](https://scikit-learn.org/stable/)
7. [nltk](https://www.nltk.org/)
8. [pandas==1.5.3](https://pandas.pydata.org/)
9. [NumPy](https://numpy.org/)
10. [matplotlib](https://matplotlib.org/)
11. [seaborn](https://seaborn.pydata.org/)
12. [Flask](https://flask.palletsprojects.com/en/2.3.x/)
13. [React](https://react.dev/)
14. [tailwindcss](https://tailwindcss.com/)
15. [docker](https://www.docker.com/products/docker-desktop/)



## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install -r requirements.txt
```

## Execution

```bash
python -m src.app
```

## Usage
Go to:
```python
http://ec2-34-239-116-11.compute-1.amazonaws.com/
```
<img src="ui.png" alt="Image" width="500"> 


 Write a review and select an algorithm. Click Analyze.


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.