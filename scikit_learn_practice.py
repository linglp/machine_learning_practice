from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# X, y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
# gnb = GaussianNB()
# y_pred = gnb.fit(X_train, y_train).predict(X_test)

# print(y_train)


corpus = ['I see a beautiful city and a brilliant people rising from this abyss.', 
'I see the lives for which I lay down my life, peaceful, useful, prosperous and happy', 
'A wonderful fact to reflect upon, that every human creature is constituted to be that profound secret and mystery to every other',
'Karishma is here', 
'Ishan is at ArborMetrix', 
'Lingling is in Ann Arbor',
'Zongs is happy',
'Burger bear was born last year in December', 
'Zongzong has never been to Chicago',
'tiger is a rabbit',
'I have named a pokemon Dickson']

corpus_test = ['It is a far, far better thing that I do, than I have ever done; it is a far, far better rest that I go to than I have ever known', 
'You have been the last dream of my soul', 
'I wish you to know that you have been the last dream of my soul',
'And yet I have had the weakness, and have still the weakness, to wish you to know with what a sudden mastery you kindled me, heap of ashes that I am, into fire', 
'Tiger was born and adopted in Canton, China', 
'Werewolf is my favorite game',
'I love studying Python and other programming languages',
'My favorite pokemon is charmander', 
'Have you ever played one-night-ultimate-werewolf before',
'I used to lived in Gregory',
'It was the best of times, it was the worst of times']

all_dataset = corpus + corpus_test

# corpus_test = ['Karishma I see a lingling a brilliant and zongs useful from this December.', 
# 'I see the lives for which I lay down my life, peaceful, rising, prosperous and happy', 
# 'A wonderful fact to reflect upon, that every human creature is constituted to be that profound secret and mystery to every other',
# 'beautiful is here', 
# 'Ishan is at ArborMetrix', 
# 'city is in Ann Arbor',
# 'people is happy',
# 'Burger bear was born last year in abyss', 
# 'Zongzong has never been to Dickson',
# 'tiger is a rabbit',
# 'I have named a pokemon Chicago']



# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(corpus)
# names_out = vectorizer.get_feature_names()
# print(names_out)
# new_array_train = X.toarray()

# print('my train array', new_array_train[-1])
# print('my train array shape', new_array_train.shape)

# vectorizer = CountVectorizer()
# X_test = vectorizer.fit_transform(corpus_test)
# names_out_test = vectorizer.get_feature_names()
# new_array_test = X_test.toarray()

# print('my test array', new_array_test.shape)
vectorizer = TfidfVectorizer(stop_words="english")
vectorizer.fit(all_dataset)
all_words = vectorizer.get_feature_names()
our_map = {i:word for i, word in enumerate(all_words)}


#training_data = vectorizer.fit(corpus)
#print('second one', len(vectorizer.get_feature_names()))

X = vectorizer.transform(corpus)
new_array_train = X.toarray()
#print(new_array_train[:3])
print(new_array_train[-1])
print(  [ our_map[index] for index, i in enumerate(new_array_train[-1]) if i > 0]  )
print('unique words', vectorizer.get_feature_names())
print('the third element of corpus', corpus[-1])



gnb = GaussianNB()
y_train = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

# Train the model using the training sets
#y_pred = gnb.fit(new_array_train, y_train).predict([new_array_test[0]]) # would still return 1

#print(y_pred)


# Can I predict random sentences? 
# Why it still returns 1 (even if that quote is no longer from tales of cities)
# what's better for the model? Having more "tests" that equals to feature 1 or feature 0? 