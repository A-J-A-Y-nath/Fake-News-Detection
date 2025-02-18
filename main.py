import pickle
import pandas as pd
import re
import string


# Load the saved Logistic Regression model
with open("logistic_model.pkl", "rb") as model_file:
    LR = pickle.load(model_file)

# Load the saved Decision tree model
with open("Decision_tree_model.pkl", "rb") as model_file:
    DT = pickle.load(model_file)
    
# Load the saved Gradient booster model
with open("GradientBooster_model.pkl", "rb") as model_file:
    GB = pickle.load(model_file)
    
# Load the saved Random forest model
with open("RandomForest_model.pkl", "rb") as model_file:
    RF = pickle.load(model_file)
    

# Load the saved TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)
    


print("Model and vectorizer loaded successfully!")



def wordopt(text):
    text = text.lower()  # Convert text to lowercase
    
    text = re.sub(r".*?", "", text)  # Remove text inside square brackets
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation but keep spaces
    text = re.sub(r'https?://\S+|www\.\S+', "", text)  # Remove URLs
    text = re.sub(r'<.*?>+', "", text)  # Remove HTML tags
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)  # Remove punctuation
    text = re.sub(r'\n', "", text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', "", text)  # Remove words containing numbers
    
    return text




def output_lable(n):
    if n==0:
        return "Fake News"
    elif n==1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = loaded_vectorizer.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    
    return "\n\n<b>LR Predicition:</b> {} \n\n<b>DT Prediction:</b> {} \n\n<b>GB Prediction:</b> {} \n\n<b>RF Prediction:</b> {}".format(output_lable(pred_LR[0]),
                                                                                                             output_lable(pred_DT[0]),
                                                                                                             output_lable(pred_GB[0]),
                                                                                                             output_lable(pred_RF[0]))
#news = str(input("Enter the news : "))
#manual_testing(news)
    
    
    
    
    