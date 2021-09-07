from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# 展示各个文章的关于tf-idf的词分布
from visual import show_tfidf


docs = [
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "I am bob",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
    "I am kitty, I like bob",
    "I do not care who like bob, but I like kitty",
    "It is coffee time, bring your cup",
]


vectorizer = TfidfVectorizer()
tf_idf = vectorizer.fit_transform(docs)
print("idf: ", [(n, idf) for idf, n in zip(
    vectorizer.idf_, vectorizer.get_feature_names())])
print("v2i: ", vectorizer.vocabulary_)


q = "I get a coffee cup"
qtf_idf = vectorizer.transform([q])
res = cosine_similarity(tf_idf, qtf_idf)
res = res.ravel().argsort()[-3:]
print("\ntop 3 docs for '{}':\n{}".format(q, [docs[i] for i in res[::-1]]))


i2v = {i: v for v, i in vectorizer.vocabulary_.items()}
# sklearn 实现的 tf_idf 表只会存储文章中 tf×idf 值为非空的词 (用于减少内存)
# 这里将 tf_idf 表转换为显示所有词的 tf×idf 值的形式
dense_tfidf = tf_idf.todense()
show_tfidf(dense_tfidf, [i2v[i] for i in range(
    dense_tfidf.shape[1])], "tfidf_sklearn_matrix")
