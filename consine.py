import math
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


lemmatizer = WordNetLemmatizer()  
stop_words = set(stopwords.words('english'))
WORD = re.compile(r"\w+")

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    words = WORD.findall(text)
    words = [lemmatizer.lemmatize(w).lower() for w in words if not w.lower() in stop_words]
    return Counter(words)


def listtocosine(reference_string,hypothesis_list):
    hypothesis_string = " ".join(hypothesis_list[1:-1])
    reference_vector = text_to_vector(reference_string)
    hypothesis_vector = text_to_vector(hypothesis_string)
    print(reference_vector)
    print(hypothesis_vector)
    cosine = get_cosine(reference_vector, hypothesis_vector)
    return cosine


if __name__ == "__main__":
    reference_string = "White dogs running"
    s1 = ['<start>', 'a', 'little', 'boy', 'in', 'a', 'white', 'shirt', 'is', 'sitting', 'in', 'the', 'grass', '<end>']
    s2 = ['<start>', 'a', 'brown', 'dog', 'is', 'running', 'in', 'the', 'sand', '<end>']
    s3 = ['<start>', 'a', 'brown', 'dog', 'runs', 'through', 'the', 'grass', '<end>']
    s4 = ['<start>', 'a', 'little', 'boy', 'in', 'a', 'striped', 'shirt', 'is', 'holding', 'a', 'spoon', '<end>']
    s5 = ['<start>', 'two', 'white', 'dogs', 'running', 'in', 'the', 'grass', '<end>']
    for i in [s1,s2,s3,s4,s5]:
        print(listtocosine(reference_string,i))  