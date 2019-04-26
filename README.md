# NBA Stats Projection Model
KNN inspired NBA Stats Fantasy Projection Model in Python with Pandas

## Getting Started
These instructions will get you a copy of the model up and running on your local machine for development and testing purposes. This is for MacOS.

### Prerequisites
Recommend using Homebrew to install python and pyenv. Need to install [xcode](https://itunes.apple.com/us/app/xcode/id497799835?mt=12) first then install [homebrew](https://brew.sh/).
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

### Installation
A step by step guide to get your venv running

**Install pyenv using homebrew on macOS**
```
brew install pyenv
```
**Install python using pyenv**
```
pyenv install 3.6.8
```
Install at the global level
```
pyenv global 3.6.6
pyenv shell 3.6.6
```
**Create a virtual environment**
```
python -m venv venv
```
Activate the virtual environment
```
source venv/bin/activate
```
**Install packages**
```
pip install -r requirements.txt
```

### CSV Files
Data pulled from [NBA stats](https://stats.nba.com/) and saved down into CSV files to be imported into dataframes. Filenames may be updated over time. 

### Acknowledgments
- While data was previously pulled down via NBA API, I came across an NBA chrome extension that made it super easy to get NBA stats in CSV form. [NBA Data Retriever](https://chrome.google.com/webstore/detail/nba-data-retriever/cibebblabkdibhnidfnipfnjkfbcmeha?hl=en)
- Projection Model was heavily inspired by the [FATS Model from NBA Math](https://nbamath.com/fats-model/)

#### Notes
I am in the process of turning this into a Udemy class focused on pandas basics but also walking through how to build the model step by step. Any feedback I can get would be extremely helpful!
